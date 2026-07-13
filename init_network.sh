#!/usr/bin/env bash
# Initialize tc-netem latency rules between fog node containers.
# Called by the tc-netem sidecar containers.
set -euo pipefail

LATENCY_MS="${LATENCY_MS:-10}"
TARGET_NET="${TARGET_NET:-192.168.100.0/24}"

# The sidecar container has direct access to the fog node's veth.
# We add a delay on egress traffic.
tc qdisc add dev eth0 root netem delay "${LATENCY_MS}ms" 2>/dev/null || \
tc qdisc change dev eth0 root netem delay "${LATENCY_MS}ms"

echo "tc-netem: added ${LATENCY_MS}ms delay on eth0 to ${TARGET_NET}"
