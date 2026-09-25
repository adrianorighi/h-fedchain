#!/usr/bin/env python3
"""Edge Client for H-FedChain distributed FL.

Registers the device's verification key, then publishes the gradient (as
float32 bytes) via MQTT to its Fog cluster after local training.

Usage:
    python -m services.edge_client --device-id d0 --cluster-id c1 \
        --fog-broker fog1.local --round 1
"""
import argparse
import asyncio
import hashlib
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network.mqtt import MqttClient
from core.pki import gradient_signed_message, sign as pki_sign
from core.pki.persistence import load_or_create_keypair
from dataset.model import MLP
from hfc_types.messages import Gradient, GradientWithProof


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H-FedChain Edge Client")
    parser.add_argument("--device-id", required=True, help="Unique device identifier")
    parser.add_argument(
        "--cluster-id",
        default=os.environ.get("CLUSTER_ID", "default"),
        help="Target Fog cluster (default: CLUSTER_ID env, else 'default')",
    )
    parser.add_argument("--fog-broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--fog-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--round", type=int, required=True, help="Current FL round")
    parser.add_argument("--use-snark", action="store_true", help="Generate SNARK proof")
    parser.add_argument("--input-dim", type=int, default=12000, help="MLP input dimension")
    parser.add_argument("--adversarial", action="store_true", help="Simulate adversarial behavior")
    parser.add_argument(
        "--key-file",
        default=None,
        help="Ed25519 key file (default: ~/.hfc/keys/edge/<cluster-id>/<device-id>.key)",
    )
    return parser.parse_args(argv)


def default_key_path(cluster_id: str, device_id: str) -> str:
    """Per-cluster persistent key path, matching edge_runner's default."""
    return os.path.join(
        os.path.expanduser("~/.hfc/keys/edge"), cluster_id, f"{device_id}.key"
    )


def load_keys(
    key_file: str | None, cluster_id: str, device_id: str
) -> tuple[bytes, bytes]:
    """Load (or create on first run) the device's persistent keypair."""
    return load_or_create_keypair(key_file or default_key_path(cluster_id, device_id))


def compute_flat_gradient(
    input_dim: int, adversarial: bool = False
) -> tuple[np.ndarray, dict]:
    """Random-weight demo gradient. Returns (flat_grad, global_weights)."""
    model = MLP(input_dim)
    rng = np.random.default_rng()
    global_weights = {
        "W1": rng.standard_normal((input_dim, 64)).astype(np.float32),
        "b1": rng.standard_normal(64).astype(np.float32),
        "W2": rng.standard_normal((64, 5)).astype(np.float32),
        "b2": rng.standard_normal(5).astype(np.float32),
    }

    X = rng.standard_normal((32, input_dim)).astype(np.float32)
    y = rng.integers(0, 5, size=(32,)).astype(np.int64)
    flat_grad = model.compute_gradient(X, y, global_weights)

    if adversarial:
        flat_grad = -flat_grad
    return flat_grad, global_weights


def build_payloads(
    device_id: str,
    cluster_id: str,
    round_num: int,
    flat_grad,
    sk: bytes,
    vk: bytes,
    snark_proof=None,
) -> tuple[str, bytes, str, bytes]:
    """Build the registration and gradient MQTT payloads (no broker needed).

    Returns (reg_topic, reg_payload, grad_topic, grad_payload). The gradient
    travels as raw float32 bytes (the compact wire form) and is signed over
    that canonical form, so verification succeeds on the published message.
    """
    data_bytes = np.asarray(flat_grad, dtype=np.float32).tobytes()
    signature = pki_sign(
        sk, gradient_signed_message(device_id, round_num, data_bytes)
    )

    gradient = Gradient(
        node_id=device_id,
        round=round_num,
        data=data_bytes,
        signature=signature,
    )
    gradient_with_proof = GradientWithProof(
        gradient=gradient, snark_proof=snark_proof
    )

    reg_topic = f"register/{cluster_id}/{device_id}"
    reg_payload = pickle.dumps({"node_id": device_id, "vk": vk})
    grad_topic = f"gradients/{cluster_id}"
    grad_payload = pickle.dumps(gradient_with_proof)
    return reg_topic, reg_payload, grad_topic, grad_payload


async def _generate_snark(
    device_id: str, round_num: int, flat_grad: np.ndarray, sk: bytes, model_hash: bytes
):
    try:
        from zkp.snark import SnarkProver

        # The prover hashes gradient.data as given; feed it the list form so
        # np.linalg.norm in the prover sees numbers, not raw bytes.
        gradient = Gradient(
            node_id=device_id, round=round_num, data=flat_grad.tolist()
        )
        prover = SnarkProver()
        proof = await prover.generate_proof(gradient, model_hash, sk)
        print(f"[{device_id}] SNARK proof generated ({len(proof.proof_bytes)} bytes)")
        return proof
    except Exception as e:
        print(f"[{device_id}] SNARK generation failed: {e}")
        return None


async def main(argv: list[str] | None = None, mqtt_factory=None):
    args = parse_args(argv)

    sk, vk = load_keys(args.key_file, args.cluster_id, args.device_id)

    flat_grad, global_weights = compute_flat_gradient(
        args.input_dim, adversarial=args.adversarial
    )

    snark_proof = None
    if args.use_snark:
        model_hash = hashlib.sha256(
            str(sorted(global_weights.items())).encode()
        ).digest()
        snark_proof = await _generate_snark(
            args.device_id, args.round, flat_grad, sk, model_hash
        )

    reg_topic, reg_payload, grad_topic, grad_payload = build_payloads(
        args.device_id,
        args.cluster_id,
        args.round,
        flat_grad,
        sk,
        vk,
        snark_proof=snark_proof,
    )

    factory = mqtt_factory or MqttClient
    client = factory(args.device_id, args.fog_broker, args.fog_port)
    await client.start()

    # Registration must land before the gradient: the fog's PKI gate rejects
    # gradients from unknown devices.
    if not client.publish(reg_topic, reg_payload, retain=True):
        print(
            f"[{args.device_id}] WARNING: registration not delivered to {reg_topic}"
        )
    print(f"[{args.device_id}] Registered vk on {reg_topic}")

    if not client.publish(grad_topic, grad_payload):
        print(
            f"[{args.device_id}] WARNING: gradient not delivered to {grad_topic}"
        )
    print(
        f"[{args.device_id}] Gradient published to {grad_topic} "
        f"({len(grad_payload)} bytes)"
    )

    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
