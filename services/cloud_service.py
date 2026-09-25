"""Cloud service — top-tier gRPC aggregation node for H-FedChain."""

import asyncio
import hashlib
import logging
import pickle
import socket
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.cloud.model_validation import ModelValidationGate
from core.ledger.worm_store import WormStore
from core.network.mqtt import MqttClient
from hfc_types.block import Block, GlobalOutput
from hfc_types.crypto import StarkProof
from zkp.stark import StarkProver, StarkVerifier
from services.identity_service import GlobalIdentityService

if TYPE_CHECKING:
    from proto import hfedchain_pb2

logger = logging.getLogger(__name__)


class CloudService:
    def __init__(
        self,
        n_expected_clusters: int = 1,
        grpc_port: int = 50052,
        learning_rate: float = 0.01,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_enabled: bool = True,
        worm_db_path: Optional[str] = None,
    ):
        self.n_expected_clusters = n_expected_clusters
        self.grpc_port = grpc_port
        self.learning_rate = learning_rate

        self.validation_gate = ModelValidationGate()
        self.worm = WormStore(db_path=worm_db_path)
        self.stark_verifier = StarkVerifier()
        self.stark_prover = StarkProver()
        self.identity_service = GlobalIdentityService()
        self.ca_vk = self.identity_service.ca_vk
        self.converged = False
        self.global_weights: Optional[dict] = None

        # Unique client_id per instance (pod hostname): a rolling restart
        # overlaps old+new pods for a few seconds and mosquitto force-closes
        # the old session on a duplicate id ("session taken over") — with a
        # shared id the two pods kick each other in a ~1/s ping-pong and a
        # broadcast landing in that window is dropped.
        self.mqtt = MqttClient(
            f"cloud-{socket.gethostname()}", mqtt_broker, mqtt_port
        )
        self._mqtt_enabled = mqtt_enabled
        self._last_global_hash = b"\x00" * 32

        self._grpc_server = None
        self._round_outputs: dict[int, dict[str, dict]] = {}
        # Rounds whose aggregation already ran (accepted OR gate-rejected):
        # a fog retry must rebuild the response, never re-aggregate/fork.
        self._aggregated_rounds: set[int] = set()
        # Aggregation outcome kept in memory so gate-rejected rounds (which
        # are still appended to WORM as compliance records) can answer
        # retries even if the ledger append is unavailable.
        self._aggregated_results: dict[int, dict] = {}

    async def start(self):
        import grpc
        from proto import hfedchain_pb2_grpc

        if self._mqtt_enabled:
            try:
                await self.mqtt.start()
            except Exception as e:
                logger.error("MQTT connection failed for cloud: %s", e)
        self._grpc_server = grpc.aio.server()
        hfedchain_pb2_grpc.add_CloudAggregationServicer_to_server(
            CloudAggregationServicer(self), self._grpc_server,
        )
        self._grpc_server.add_insecure_port(f"0.0.0.0:{self.grpc_port}")
        await         self._grpc_server.start()
        print(f"[cloud] gRPC server listening on :{self.grpc_port}", flush=True)
        await self._grpc_server.wait_for_termination()

    async def stop(self):
        if self._grpc_server:
            await self._grpc_server.stop(None)
        if self._mqtt_enabled:
            try:
                await self.mqtt.stop()
            except Exception as e:
                logger.error("MQTT stop failed for cloud: %s", e)

    async def submit_cluster_output(
        self, cluster_output_proto
    ) -> "hfedchain_pb2.GlobalOutputMessage":
        from proto import hfedchain_pb2

        round_num = cluster_output_proto.round
        cluster_id = cluster_output_proto.cluster_id

        # End-to-end delta integrity: the fog binds sha256(delta_w) into the
        # committed block's gradient_hash; a payload that does not hash to it
        # (e.g. a re-drive that shipped another round's delta) is dropped for
        # this round and never counted in n_active/weights.
        gradient_hash = cluster_output_proto.gradient_hash
        if gradient_hash:
            if (
                hashlib.sha256(cluster_output_proto.delta_w).digest()
                != gradient_hash
            ):
                logger.warning(
                    "Dropping cluster %s output for round %d: delta_w does "
                    "not match committed gradient_hash",
                    cluster_id, round_num,
                )
                return hfedchain_pb2.GlobalOutputMessage(
                    delta_w_inter=b"", n_active_clusters=0, round=round_num,
                    converged=False,
                )
        else:
            # proto3 default: legacy/fog-less producers skip the check.
            logger.debug(
                "Cluster %s output for round %d has empty gradient_hash; "
                "skipping integrity check", cluster_id, round_num,
            )

        # Idempotent re-submit: a fog retry of an already-aggregated round
        # (accepted or gate-rejected) gets the recorded response back instead
        # of restarting collection for a finished round — re-aggregating would
        # mint a second pi_inter with a different prev_hash, broadcast again
        # and fork the chain. Keyed on _aggregated_rounds, not WORM presence,
        # because a gate-rejected round has its own (accepted=False) WORM
        # record but must be answerable even without one.
        if round_num in self._aggregated_rounds and round_num not in self._round_outputs:
            stored = self.worm.get_entry(round_num)
            if stored is not None:
                return hfedchain_pb2.GlobalOutputMessage(
                    delta_w_inter=stored.delta_w_inter,
                    pi_inter=stored.pi_inter.proof_bytes
                    if stored.pi_inter else None,
                    n_active_clusters=stored.n_active_clusters,
                    round=stored.round_num,
                    converged=self.converged,
                )
            cached = self._aggregated_results.get(round_num)
            if cached is not None:
                return hfedchain_pb2.GlobalOutputMessage(
                    delta_w_inter=cached["delta_w_inter"],
                    pi_inter=cached["pi_inter"].proof_bytes
                    if cached["pi_inter"] else None,
                    n_active_clusters=cached["n_active_clusters"],
                    round=round_num,
                    converged=cached["converged"],
                )
            # Marked but neither persisted nor finished (a retry arrived
            # while the proof was still being generated): the in-flight call
            # owns the round; report nothing yet rather than re-collecting.
            logger.debug(
                "Round %d aggregation still in flight; deferring retry",
                round_num,
            )
            return hfedchain_pb2.GlobalOutputMessage(
                delta_w_inter=b"", n_active_clusters=0, round=round_num,
                converged=False,
            )

        # Verify STARK proof if present
        if cluster_output_proto.stark_proof:
            proof = StarkProof(
                proof_bytes=cluster_output_proto.stark_proof,
                public_inputs={"round": round_num, "cluster_id": cluster_id},
            )
            if not await self.stark_verifier.verify(proof, proof.public_inputs):
                return hfedchain_pb2.GlobalOutputMessage(
                    delta_w_inter=b"", n_active_clusters=0, round=round_num,
                    converged=False,
                )

        # Collect outputs per round, keyed by cluster_id
        if round_num not in self._round_outputs:
            self._round_outputs[round_num] = {}
        self._round_outputs[round_num][cluster_id] = {
            "delta_w": cluster_output_proto.delta_w,
            "n_devices": cluster_output_proto.n_devices,
        }
        print(
            f"[cloud] received cluster={cluster_id} round={round_num} "
            f"({len(self._round_outputs[round_num])}/{self.n_expected_clusters})",
            flush=True,
        )

        # Aggregate once all clusters have reported
        if len(self._round_outputs[round_num]) >= self.n_expected_clusters:
            return await self._aggregate_round(round_num)

        return hfedchain_pb2.GlobalOutputMessage(
            delta_w_inter=b"", n_active_clusters=0, round=round_num,
            converged=False,
        )

    async def _aggregate_round(self, round_num: int):
        from proto import hfedchain_pb2

        round_data = self._round_outputs.pop(round_num)
        # Claim the round before the first await: a retry arriving while the
        # STARK proof is being generated must hit the idempotency guard above
        # instead of re-collecting and re-aggregating this round.
        self._aggregated_rounds.add(round_num)
        cluster_ids = sorted(round_data)
        outputs = list(round_data.values())
        n_active = len(outputs)

        # Weighted FedAvg over non-empty deltas: clusters reporting an empty
        # delta are excluded from both arrays and weights, and the weights
        # are normalized over the included ones so they sum to 1 (a skipped
        # cluster must not shrink the remaining contribution).
        arrays = []
        weights = []
        included_devices = sum(o["n_devices"] for o in outputs if o["delta_w"])
        for o in outputs:
            if o["delta_w"]:
                arrays.append(np.frombuffer(o["delta_w"], dtype=np.float32))
                weights.append(
                    o["n_devices"] / included_devices
                    if included_devices > 0 else 1.0
                )

        if arrays:
            w_sum = sum(weights)
            if w_sum > 0:
                weights = [w / w_sum for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            weighted = sum(w * a for w, a in zip(weights, arrays))
            delta_w_inter = weighted.astype(np.float32).tobytes()
        else:
            weighted = np.zeros(1, dtype=np.float32)
            delta_w_inter = b""

        # (a) Validation gate on the cloud's own mirror: no labels here, so
        # loss stays None and only the delta_norm criterion applies. The
        # mirror starts as zeros of the delta's shape on the first round.
        if self.global_weights is None:
            self.global_weights = {
                "global": np.zeros(len(weighted), dtype=np.float32)
            }
        w_new = {
            "global": self.global_weights["global"]
            - self.learning_rate * weighted
        }
        result = self.validation_gate.validate(self.global_weights, w_new)
        if result.accepted:
            self.global_weights = w_new
        else:
            # Rejected: the mirror stays put, but the round is still
            # recorded (accepted=False) and broadcast below — edge devices
            # must keep progressing even when the cloud refuses the update.
            logger.warning(
                "Validation gate rejected round %d (%s); recording the "
                "rejection in WORM and still broadcasting",
                round_num, result.reason,
            )

        # (b) Inter-regional STARK proof over the weighted delta, hash-chained
        # to the previous global block.
        inter_block = Block(
            round=round_num,
            gradient_hash=hashlib.sha256(delta_w_inter).digest(),
            qc_commit=None,
            stark_proof=None,
            accepted_devices=cluster_ids,
            rejected_devices=[],
            timestamp=time.time(),
            prev_hash=self._last_global_hash,
            n=self.n_expected_clusters,
            f=0,
        )
        pi_inter = await self.stark_prover.generate_proof(inter_block)
        if not await self.stark_verifier.verify(
            pi_inter, pi_inter.public_inputs
        ):
            # Same pattern as the fog's regional path: the proof travels and
            # the receiving verifier decides.
            logger.warning(
                "Self-verification of pi_inter failed for round %d; "
                "forwarding the proof anyway", round_num,
            )
        self._last_global_hash = inter_block.hash

        # Record the outcome for idempotent retries (kept alongside the WORM
        # entry so a rejected round is answerable even if the append fails).
        self._aggregated_results[round_num] = {
            "delta_w_inter": delta_w_inter,
            "pi_inter": pi_inter,
            "n_active_clusters": n_active,
            "converged": self.converged,
        }

        # (c) Persist the round to the WORM ledger — accepted rounds as the
        # canonical record, rejected ones as compliance evidence that the
        # gate saw and refused them (accepted=False).
        appended = self.worm.append(GlobalOutput(
            delta_w_inter=delta_w_inter,
            pi_inter=pi_inter,
            n_active_clusters=n_active,
            round_num=round_num,
            accepted=result.accepted,
        ))
        if not appended:
            logger.warning(
                "WORM append for round %d was a duplicate", round_num
            )

        # (d) Broadcast the delta (not full weights) for the edges to apply
        # w -= lr * delta locally with the same learning_rate; a broker
        # outage must not fail the RPC.
        broadcast_ok = False
        if self._mqtt_enabled:
            payload = {
                "schema": 1,
                "round": round_num,
                "delta_w_inter": delta_w_inter,
                # sha256(delta), NOT sha256(weights) — that meaning is
                # reserved for model_hash elsewhere, so the delta gets its
                # own key instead of colliding with it.
                "delta_hash": hashlib.sha256(delta_w_inter).digest(),
                "learning_rate": self.learning_rate,
                "pi_inter": pi_inter.proof_bytes,
                "n_active_clusters": n_active,
                "converged": self.converged,
            }
            try:
                delivered = self.mqtt.publish(
                    "models/global", pickle.dumps(payload), retain=True
                )
            except Exception as e:
                logger.error("Global model broadcast failed: %s", e)
            else:
                if not delivered:
                    logger.warning(
                        "global model for round %d not delivered", round_num
                    )
                else:
                    broadcast_ok = True

        # Per-round flushed evidence of the aggregate+broadcast pair (the
        # local e2e scripts/e2e_local.sh greps this line for rounds 1..N).
        print(
            f"[cloud] round {round_num} aggregated "
            f"clusters={n_active}/{self.n_expected_clusters} "
            f"accepted={result.accepted} "
            f"broadcast={'yes' if broadcast_ok else 'no'}",
            flush=True,
        )

        # (e) Response to the last reporting fog.
        return hfedchain_pb2.GlobalOutputMessage(
            delta_w_inter=delta_w_inter,
            pi_inter=pi_inter.proof_bytes,
            n_active_clusters=n_active,
            round=round_num,
            converged=self.converged,
        )


class CloudAggregationServicer:
    def __init__(self, cloud: CloudService):
        self.cloud = cloud

    async def SubmitClusterOutput(self, request, context):
        return await self.cloud.submit_cluster_output(request)


if __name__ == "__main__":
    # `python -m services.cloud_service` — same environment-driven startup as
    # `python -m services.entrypoint` with SERVICE=cloud.
    import os

    from services.entrypoint import main as entrypoint_main

    os.environ.setdefault("SERVICE", "cloud")
    asyncio.run(entrypoint_main())
