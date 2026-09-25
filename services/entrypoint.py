"""Entrypoint for Docker containers: starts Fog or Cloud service based on SERVICE env."""
import os
import asyncio
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.pki.persistence import load_or_create_keypair  # noqa: F401  (public name kept)


def parse_peers(peers_str: str, default_port: int) -> tuple[list[str], dict[str, str]]:
    """PEERS entries: 'host:port', 'host' or 'node_id@host:port'."""
    peers: list[str] = []
    addrs: dict[str, str] = {}
    for entry in [e.strip() for e in peers_str.split(",") if e.strip()]:
        if "@" in entry:
            nid, addr = entry.split("@", 1)
        elif ":" in entry:
            nid, addr = entry.split(":", 1)[0], entry
        else:
            nid, addr = entry, f"{entry}:{default_port}"
        peers.append(nid)
        addrs[nid] = addr
    return peers, addrs


def resolve_membership(n: int, f: int, peers: list[str], node_id: str) -> tuple[int, int]:
    """Reconcile the env-configured (N, F) with the actual PEERS list.

    The consensus quorum is derived from n, but the single-node commit path
    triggers on an empty peers list. With N=3 and no PEERS the engine would
    validate a 1-signature QC against quorum_size(3)=2 and reject every
    round, so membership must always be len(peers) + 1.

    f is only valid while f < n (quorum = n - f must stay >= 1, and the
    view-change threshold n - f must too); clamp it only in that case.
    """
    n_eff = len(peers) + 1
    if n != n_eff:
        print(
            f"[{node_id}] Warning: N={n} overridden by actual peer "
            f"membership (len(PEERS)+1); using n={n_eff}",
            flush=True,
        )
    if f >= n_eff:
        f_clamped = n_eff - 1
        print(
            f"[{node_id}] Warning: F={f} invalid for n={n_eff} (need f < n); "
            f"using f={f_clamped}",
            flush=True,
        )
        f = f_clamped
    return n_eff, f


def mqtt_enabled_from_env(default: str = "1") -> bool:
    """Parse MQTT_ENABLED: "0"/"false"/"no" (case-insensitive, padded) mean
    disabled; everything else (including an unset var) means enabled."""
    return os.environ.get("MQTT_ENABLED", default).strip().lower() not in {
        "0", "false", "no",
    }


def build_fog_service():
    """Build (but do not start) the FogService from the environment."""
    from services.fog_service import FogService

    node_id = os.environ.get("NODE_ID", "fog1")
    n = int(os.environ.get("N", "3"))
    f = int(os.environ.get("F", "1"))
    grpc_port = int(os.environ.get("GRPC_PORT", "50051"))
    mqtt_broker = os.environ.get("MQTT_BROKER", "mqtt-broker")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    mqtt_enabled = mqtt_enabled_from_env()
    variant = os.environ.get("VARIANT", "no_zkp")

    peers, peer_addrs = parse_peers(os.environ.get("PEERS", ""), grpc_port)
    n, f = resolve_membership(n, f, peers, node_id)
    key_path = os.environ.get("NODE_KEY_FILE") or (
        f"/var/lib/hfc/keys/{node_id}.key"
    )
    sk, vk = load_or_create_keypair(key_path)
    print(f"[{node_id}] Node key file: {key_path}", flush=True)

    return FogService(
        node_id=node_id, sk=sk, vk=vk,
        peers=peers, n=n, f=f, variant=variant,
        grpc_port=grpc_port,
        mqtt_broker=mqtt_broker, mqtt_port=mqtt_port,
        mqtt_enabled=mqtt_enabled,
        cluster_id=os.environ.get("CLUSTER_ID", "default"),
        gradient_threshold=int(os.environ.get("GRADIENT_THRESHOLD", "0")),
        cloud_address=os.environ.get("CLOUD_ADDRESS", "cloud:50052"),
        peer_addrs=peer_addrs,
        vote_timeout_s=float(os.environ.get("VOTE_TIMEOUT_S", "5.0")),
    )


def build_cloud_service():
    """Build (but do not start) the CloudService from the environment."""
    from services.cloud_service import CloudService

    grpc_port = int(os.environ.get("GRPC_PORT", "50052"))
    n_expected_clusters = int(os.environ.get("N_EXPECTED_CLUSTERS", "1"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.01"))
    mqtt_broker = os.environ.get("MQTT_BROKER", "mqtt-broker")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    mqtt_enabled = mqtt_enabled_from_env()
    worm_db_path = os.environ.get("WORM_DB_PATH") or None

    cloud = CloudService(
        n_expected_clusters=n_expected_clusters,
        grpc_port=grpc_port,
        learning_rate=learning_rate,
        mqtt_broker=mqtt_broker,
        mqtt_port=mqtt_port,
        mqtt_enabled=mqtt_enabled,
        worm_db_path=worm_db_path,
    )
    print(
        f"[cloud] config: expected_clusters={n_expected_clusters} "
        f"lr={learning_rate} grpc_port={grpc_port} "
        f"mqtt={mqtt_broker}:{mqtt_port} enabled={mqtt_enabled} "
        f"worm_db={worm_db_path}",
        flush=True,
    )
    return cloud


async def main():
    service = os.environ.get("SERVICE", "fog").lower()

    if service == "fog":
        fog = build_fog_service()
        print(
            f"[{fog.node_id}] Starting Fog service on :{fog.grpc_port} "
            f"(n={fog.n}, f={fog.f})",
            flush=True,
        )
        await fog.start()
        print(
            f"[{fog.node_id}] Fog service running on :{fog.grpc_port} "
            f"(n={fog.n}, f={fog.f}, cluster={fog.cluster_id})",
            flush=True,
        )
        await asyncio.Event().wait()  # run until cancelled (SIGTERM)

    elif service == "cloud":
        cloud = build_cloud_service()

        print(
            f"[cloud] Starting Cloud service on :{cloud.grpc_port} "
            f"(expected_clusters={cloud.n_expected_clusters})",
            flush=True,
        )
        await cloud.start()

    else:
        print(f"Unknown SERVICE: {service}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
