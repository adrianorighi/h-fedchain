#!/usr/bin/env python3
"""Edge Client for H-FedChain distributed FL.

Publishes gradient via MQTT to Fog cluster after local training.

Usage:
    python -m services.edge_client --device-id d0 --fog-broker fog1.local --round 1
"""
import argparse
import asyncio
import hashlib
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network.mqtt import MqttClient
from core.pki import generate_keypair, sign as pki_sign
from dataset.model import MLP
from hfc_types.messages import Gradient, GradientWithProof


async def main():
    parser = argparse.ArgumentParser(description="H-FedChain Edge Client")
    parser.add_argument("--device-id", required=True, help="Unique device identifier")
    parser.add_argument("--fog-broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--fog-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--round", type=int, required=True, help="Current FL round")
    parser.add_argument("--use-snark", action="store_true", help="Generate SNARK proof")
    parser.add_argument("--input-dim", type=int, default=12000, help="MLP input dimension")
    parser.add_argument("--adversarial", action="store_true", help="Simulate adversarial behavior")
    args = parser.parse_args()

    sk, vk = generate_keypair()

    model = MLP(args.input_dim)
    global_weights = {
        "W1": np.random.randn(12000, 64).astype(np.float32),
        "b1": np.random.randn(64).astype(np.float32),
        "W2": np.random.randn(64, 5).astype(np.float32),
        "b2": np.random.randn(5).astype(np.float32),
    }

    X = np.random.randn(32, args.input_dim).astype(np.float32)
    y = np.random.randint(0, 5, size=(32,)).astype(np.int64)
    flat_grad = model.compute_gradient(X, y, global_weights)

    if args.adversarial:
        flat_grad = -flat_grad

    gradient = Gradient(
        node_id=args.device_id,
        round=args.round,
        data=flat_grad.tolist(),
        signature=pki_sign(sk, flat_grad.tobytes()),
    )

    gradient_with_proof = GradientWithProof(gradient=gradient, snark_proof=None)

    if args.use_snark:
        try:
            from zkp.snark import SnarkProver

            model_hash = hashlib.sha256(
                str(sorted(global_weights.items())).encode()
            ).digest()
            prover = SnarkProver()
            proof = await prover.generate_proof(gradient, model_hash, sk)
            gradient_with_proof.snark_proof = proof
            print(f"[{args.device_id}] SNARK proof generated ({len(proof.proof_bytes)} bytes)")
        except Exception as e:
            print(f"[{args.device_id}] SNARK generation failed: {e}")

    client = MqttClient(args.device_id, args.fog_broker, args.fog_port)
    await client.start()

    payload = pickle.dumps(gradient_with_proof)
    topic = f"gradients/default"
    client.publish(topic, payload)
    print(f"[{args.device_id}] Gradient published to {topic} ({len(payload)} bytes)")

    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
