"""Monolithic edge runner for the H-FedChain distributed pipeline.

Owns the full device-side loop: PTB-XL (or synthetic) data load, Dirichlet
partition, persistent per-device keys, MQTT registration, gradient upload,
global-model wait/validate/apply.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import pickle
import time

import numpy as np

from core.network.mqtt import MqttClient
from core.pki.persistence import load_or_create_keypair
from dataset.edge_worker import EdgeWorker
from dataset.loader import PTBXLLoader
from dataset.model import MLP, flatten_weights, unflatten_weights
from dataset.partitioner import DirichletPartitioner
from hfc_types.crypto import StarkProof
from simulator.orchestrator import select_adversarial_ids
from zkp.stark import StarkVerifier

logger = logging.getLogger(__name__)

WAIT_SLICE_S = 0.5
SEED = 42


class EdgeRunner:
    def __init__(
        self,
        device_ids: list[str] | None = None,
        num_devices: int = 10,
        cluster_id: str = "default",
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        rounds: int = 10,
        learning_rate: float = 0.01,
        use_snark: bool = False,
        use_dataset: bool = False,
        key_dir: str | None = None,
        model_timeout_s: float = 60.0,
        dirichlet_alpha: float = 0.5,
        max_records: int = 500,
        adversarial_ratio: float = 0.0,
        input_dim: int = 12000,
        republish_interval_s: float = 15.0,
        mqtt_client: MqttClient | None = None,
    ):
        if device_ids is not None:
            self.device_ids: list[str] | None = list(device_ids)
            self.num_devices = len(self.device_ids)
        else:
            self.device_ids = None
            self.num_devices = num_devices
        self.cluster_id = cluster_id
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.use_snark = use_snark
        self.use_dataset = use_dataset
        if key_dir:
            self.key_dir = key_dir
        else:
            self.key_dir = os.environ.get("KEY_DIR") or os.path.join(
                os.path.expanduser("~/.hfc/keys/edge"), cluster_id
            )
        self.model_timeout_s = model_timeout_s
        self.dirichlet_alpha = dirichlet_alpha
        self.max_records = max_records
        self.adversarial_ratio = adversarial_ratio
        self.input_dim = input_dim
        self.republish_interval_s = republish_interval_s
        # per-cluster partition/data seed; model init keeps SEED (globally shared)
        self.partition_seed = int.from_bytes(
            hashlib.sha256(cluster_id.encode()).digest()[:4], "big"
        ) % (2 ** 31)

        self.mqtt: MqttClient | None = mqtt_client
        self.workers: list[EdgeWorker] = []
        self.global_weights: dict | None = None
        self.rounds_completed = 0
        self.start_round = 1

        self._current_round = 0
        self._model_event = asyncio.Event()
        self._last_model: dict | None = None
        self._model_buffer: dict[int, dict] = {}
        self._mqtt_started = False

    # ------------------------------------------------------------------
    # Persisted progress (retained models/global must not cross runs)
    # ------------------------------------------------------------------

    @property
    def _progress_path(self) -> str:
        return os.path.join(self.key_dir, f"{self.cluster_id}.progress")

    def _load_progress(self) -> int | None:
        """Last COMPLETED round of a previous run, or None when absent/unreadable."""
        try:
            with open(self._progress_path) as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[edge] unreadable progress file %s: %s",
                           self._progress_path, exc)
            return None
        last = data.get("last_completed") if isinstance(data, dict) else None
        if not isinstance(last, int) or isinstance(last, bool):
            logger.warning("[edge] progress file %s has no usable "
                           "last_completed: %r", self._progress_path, last)
            return None
        return last

    def _save_progress(self, last_completed: int) -> None:
        try:
            os.makedirs(self.key_dir, exist_ok=True)
            tmp = self._progress_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"last_completed": last_completed}, fh)
            os.replace(tmp, self._progress_path)
        except OSError as exc:
            logger.warning("[edge] could not persist progress to %s: %s",
                           self._progress_path, exc)

    def _resume_start_round(self) -> int:
        last = self._load_progress()
        if last is None:
            return 1
        if last >= self.rounds:
            logger.info(
                "[edge] previous run already finished (last_completed=%d, "
                "rounds=%d); restarting at round 1", last, self.rounds,
            )
            return 1
        logger.info("[edge] resuming at round %d", last + 1)
        return last + 1

    # ------------------------------------------------------------------
    # Setup: data, partition, keys, MQTT, registration
    # ------------------------------------------------------------------

    async def setup(self):
        self.start_round = self._resume_start_round()
        if self.device_ids is None:
            adv_ids = select_adversarial_ids(
                self.num_devices, self.adversarial_ratio, SEED
            )
            self.device_ids = [
                f"adv_{i}" if i in adv_ids else f"d{i}"
                for i in range(self.num_devices)
            ]
        else:
            adv_ids = select_adversarial_ids(
                len(self.device_ids), self.adversarial_ratio, SEED
            )

        data, labels = await self._load_data()

        partitioner = DirichletPartitioner(
            alpha=self.dirichlet_alpha, seed=self.partition_seed
        )
        assignments = partitioner.assign(
            num_devices=len(self.device_ids), labels=labels, num_classes=5
        )

        os.makedirs(self.key_dir, exist_ok=True)
        self.workers = []
        for i, device_id in enumerate(self.device_ids):
            indices = assignments[i]
            if not indices:
                raise ValueError(
                    f"device {device_id} got an empty data partition "
                    f"(devices={len(self.device_ids)}, max_records={self.max_records}); "
                    "increase MAX_RECORDS or reduce NUM_DEVICES"
                )
            keypair = load_or_create_keypair(
                os.path.join(self.key_dir, f"{device_id}.key")
            )
            self.workers.append(EdgeWorker(
                device_id=device_id,
                indices=indices,
                all_data=data,
                all_labels=labels,
                is_adversarial=i in adv_ids,
                attack_type="label_flip",
                input_dim=self.input_dim,
                use_snark=self.use_snark,
                keypair=keypair,
                forge_signature=False,
            ))

        ref = MLP(input_dim=self.input_dim, seed=SEED)
        self.global_weights = ref.get_weights()

        if self.mqtt is None:
            self.mqtt = MqttClient(
                f"edge-{self.cluster_id}", self.mqtt_broker, self.mqtt_port
            )
        await self.mqtt.start()
        self._mqtt_started = True
        self.mqtt.subscribe("models/global", self._on_model)

        register_prefix = f"register/{self.cluster_id}"
        for worker in self.workers:
            register_topic = f"{register_prefix}/{worker.device_id}"
            payload = pickle.dumps({"node_id": worker.device_id, "vk": worker.vk})
            if not self.mqtt.publish(register_topic, payload, retain=True):
                logger.warning(
                    "registration for %s not delivered to %s",
                    worker.device_id, register_topic,
                )

    async def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        loader = PTBXLLoader(max_records=self.max_records)
        if not self.use_dataset:
            return loader.generate_synthetic(
                self.max_records, seed=self.partition_seed
            )
        if not loader.is_available():
            print(
                f"[edge] PTB-XL not cached at {loader.cache_dir}; downloading "
                "(this can take a while)...",
                flush=True,
            )
            try:
                await asyncio.to_thread(loader.download)
            except Exception as exc:
                raise RuntimeError(
                    f"PTB-XL download failed: {exc}. Download it once "
                    "(dataset/loader.py::PTBXLLoader.download) or run with "
                    "DATASET=synthetic"
                ) from exc
            print("[edge] PTB-XL download complete", flush=True)
        return await asyncio.to_thread(loader.load)

    # ------------------------------------------------------------------
    # MQTT model ingest
    # ------------------------------------------------------------------

    def _on_model(self, payload: bytes):
        try:
            msg = pickle.loads(payload)
        except Exception as exc:
            logger.error("unpickle of models/global payload failed: %s", exc)
            return
        if not isinstance(msg, dict):
            logger.error("malformed models/global payload (not a dict)")
            return
        round_num = msg.get("round")
        if not isinstance(round_num, int) or round_num < self._current_round:
            logger.info(
                "ignoring models/global for round %s (waiting for round %s)",
                round_num, self._current_round,
            )
            return
        self._prune_model_buffer()
        self._model_buffer[round_num] = msg
        if round_num == self._current_round:
            self._last_model = msg
            self._model_event.set()

    def _prune_model_buffer(self):
        """Drop buffered models from rounds the runner already moved past."""
        for stale in [r for r in self._model_buffer
                      if r < self._current_round]:
            del self._model_buffer[stale]

    # ------------------------------------------------------------------
    # Round loop
    # ------------------------------------------------------------------

    async def run(self):
        self.rounds_completed = 0
        gradient_topic = f"gradients/{self.cluster_id}"
        for r in range(self.start_round, self.rounds + 1):
            self._current_round = r
            self._model_event.clear()
            self._last_model = None
            self._prune_model_buffer()

            gradients = await asyncio.to_thread(self._train_batch, r)
            for gwp in gradients:
                gwp.gradient.data = np.asarray(
                    gwp.gradient.data, dtype=np.float32
                ).tobytes()
            pending = [(gradient_topic, pickle.dumps(g)) for g in gradients]
            for topic, payload in pending:
                if not self.mqtt.publish(topic, payload):
                    logger.warning(
                        "gradient for round %d not delivered to %s", r, topic
                    )

            msg = await self._await_model(r, pending)

            converged = await self._handle_round_update(r, msg)
            # decision made (applied or deliberately skipped): the run moves
            # on, so round r counts as completed for resume purposes
            self._save_progress(r)
            if converged:
                break

    async def _handle_round_update(self, r: int, msg: dict) -> bool:
        """Apply (or deliberately skip) round ``r``'s global update.

        Returns True when the run must stop (the cloud signalled convergence).
        """
        delta = msg.get("delta_w_inter")
        if not isinstance(delta, (bytes, bytearray)):
            logger.error(
                "[edge] round %d: delta_w_inter missing or not bytes; "
                "skipping update", r,
            )
            return False
        delta = bytes(delta)
        if hashlib.sha256(delta).digest() != msg.get("delta_hash"):
            logger.error(
                "[edge] round %d: delta_hash mismatch; skipping update", r
            )
            return False

        if msg.get("pi_inter"):
            if not await self._verify_pi_inter(r, msg["pi_inter"]):
                return False

        lr = float(msg.get("learning_rate", self.learning_rate))
        try:
            flat = flatten_weights(self.global_weights)
            step = np.frombuffer(delta, dtype=np.float32).astype(np.float64)
            updated = (flat - lr * step).astype(np.float32)
            self.global_weights = unflatten_weights(
                updated, like=self.global_weights
            )
        except ValueError as exc:
            logger.error(
                "[edge] round %d: invalid delta (%s); skipping update", r, exc
            )
            return False

        self.rounds_completed = r
        print(
            f"[edge] round {r}/{self.rounds} applied "
            f"(lr={lr}, n_active_clusters={msg.get('n_active_clusters')})",
            flush=True,
        )
        if msg.get("converged"):
            print(f"[edge] converged at round {r}; stopping", flush=True)
            return True
        return False

    def _train_batch(self, r: int) -> list:
        return [w.train_round(self.global_weights, r) for w in self.workers]

    async def _await_model(self, r: int, pending: list) -> dict:
        buffered = self._model_buffer.pop(r, None)
        if buffered is not None:
            return buffered
        deadline = time.monotonic() + self.model_timeout_s
        last_republish = time.monotonic()
        while True:
            if self._model_event.is_set():
                return self._last_model
            now = time.monotonic()
            if now - last_republish >= self.republish_interval_s:
                for topic, payload in pending:
                    if not self.mqtt.publish(topic, payload):
                        logger.warning(
                            "gradient re-publish for round %d not delivered "
                            "to %s", r, topic,
                        )
                last_republish = time.monotonic()
                now = last_republish
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError(
                    f"global model for round {r} not received within "
                    f"{self.model_timeout_s}s"
                )
            try:
                await asyncio.wait_for(
                    self._model_event.wait(),
                    timeout=min(WAIT_SLICE_S, remaining),
                )
            except asyncio.TimeoutError:
                continue

    async def _verify_pi_inter(self, r: int, proof_bytes) -> bool:
        try:
            proof = StarkProof(proof_bytes=proof_bytes, public_inputs={})
            ok = await StarkVerifier().verify(proof, proof.public_inputs)
        except Exception as exc:
            logger.error(
                "[edge] round %d: pi_inter verification error (%s); "
                "skipping update", r, exc,
            )
            return False
        if not ok:
            logger.error(
                "[edge] round %d: pi_inter verification failed; skipping update", r
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def stop(self):
        if self._mqtt_started and self.mqtt is not None:
            await self.mqtt.stop()
            self._mqtt_started = False
        self._print_summary()

    def _print_summary(self):
        if not self.workers or self.global_weights is None:
            print("[edge] summary: no devices set up", flush=True)
            return
        w0 = self.workers[0]
        w0.model.set_weights(self.global_weights)
        loss = w0.model.loss(w0.X, w0.y)
        print(
            f"[edge] summary: rounds_completed={self.rounds_completed} "
            f"devices={len(self.workers)} loss_d0={loss:.4f}",
            flush=True,
        )


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


async def _run_until_done(runner: EdgeRunner):
    try:
        await runner.setup()
        await runner.run()
    finally:
        await runner.stop()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="H-FedChain monolithic edge runner")
    parser.add_argument(
        "--device-ids", default=os.environ.get("DEVICE_IDS", ""),
        help="comma-separated device ids (default: d0..dN-1)",
    )
    parser.add_argument("--num-devices", type=int, default=_env_int("NUM_DEVICES", 10))
    parser.add_argument("--cluster-id", default=os.environ.get("CLUSTER_ID", "default"))
    parser.add_argument("--mqtt-broker", default=os.environ.get("MQTT_BROKER", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=_env_int("MQTT_PORT", 1883))
    parser.add_argument("--rounds", type=int, default=_env_int("ROUNDS", 10))
    parser.add_argument("--learning-rate", type=float,
                        default=_env_float("LEARNING_RATE", 0.01))
    parser.add_argument("--variant", choices=("no_zkp", "snark", "stark", "full"),
                        default=os.environ.get("VARIANT", "no_zkp"))
    parser.add_argument("--dataset", choices=("real", "synthetic"),
                        default=os.environ.get("DATASET", "synthetic"))
    parser.add_argument("--key-dir",
                        default=os.environ.get("KEY_DIR", ""),
                        help="key directory (default: KEY_DIR env if set, "
                             "else ~/.hfc/keys/edge/<cluster-id>)")
    parser.add_argument("--model-timeout-s", type=float,
                        default=_env_float("MODEL_TIMEOUT_S", 60.0))
    parser.add_argument("--alpha", type=float, default=_env_float("ALPHA", 0.5))
    parser.add_argument("--max-records", type=int,
                        default=_env_int("MAX_RECORDS", 500))
    parser.add_argument("--adversarial-ratio", type=float,
                        default=_env_float("ADVERSARIAL_RATIO", 0.0))
    args = parser.parse_args(argv)

    device_ids = [d.strip() for d in args.device_ids.split(",") if d.strip()] or None
    n_devices = len(device_ids) if device_ids else args.num_devices
    runner = EdgeRunner(
        device_ids=device_ids,
        num_devices=args.num_devices,
        cluster_id=args.cluster_id,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        rounds=args.rounds,
        learning_rate=args.learning_rate,
        use_snark=args.variant in ("snark", "full"),
        use_dataset=args.dataset == "real",
        key_dir=args.key_dir or None,
        model_timeout_s=args.model_timeout_s,
        dirichlet_alpha=args.alpha,
        max_records=args.max_records,
        adversarial_ratio=args.adversarial_ratio,
    )
    print(
        f"[edge] config: cluster={runner.cluster_id} devices={n_devices} "
        f"rounds={runner.rounds} lr={runner.learning_rate} "
        f"variant={args.variant} dataset={args.dataset} "
        f"key_dir={runner.key_dir} timeout={runner.model_timeout_s}s "
        f"alpha={runner.dirichlet_alpha} max_records={runner.max_records} "
        f"adv_ratio={runner.adversarial_ratio} "
        f"broker={runner.mqtt_broker}:{runner.mqtt_port}",
        flush=True,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_run_until_done(runner))


if __name__ == "__main__":
    main()
