"""Entrypoint for Docker containers: starts Fog or Cloud service based on SERVICE env."""
import os
import asyncio
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    service = os.environ.get("SERVICE", "fog").lower()

    if service == "fog":
        from services.fog_service import FogService
        from core.pki import generate_keypair

        node_id = os.environ.get("NODE_ID", "fog1")
        n = int(os.environ.get("N", "3"))
        f = int(os.environ.get("F", "1"))
        grpc_port = int(os.environ.get("GRPC_PORT", "50051"))
        mqtt_broker = os.environ.get("MQTT_BROKER", "mqtt-broker")
        mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
        variant = os.environ.get("VARIANT", "no_zkp")
        peers_str = os.environ.get("PEERS", "")

        peers = [p.split(":")[0] for p in peers_str.split(",") if p]
        sk, vk = generate_keypair()

        fog = FogService(
            node_id=node_id, sk=sk, vk=vk,
            peers=peers, n=n, f=f, variant=variant,
            grpc_port=grpc_port,
            mqtt_broker=mqtt_broker, mqtt_port=mqtt_port,
        )
        print(f"[{node_id}] Starting Fog service on :{grpc_port} (n={n}, f={f})")
        await fog.start()

    elif service == "cloud":
        from services.cloud_service import CloudService

        grpc_port = int(os.environ.get("GRPC_PORT", "50052"))
        cloud = CloudService(grpc_port=grpc_port)

        print(f"[cloud] Starting Cloud service on :{grpc_port}")
        await cloud.start()

    else:
        print(f"Unknown SERVICE: {service}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
