#!/bin/bash
# scripts/tc-netem.sh — Configura latência/jitter via tc-netem
# Uso: ./tc-netem.sh <interface> <latency_ms> <jitter_ms> [loss_percent]
#
# Exemplos:
#   ./tc-netem.sh eth0 10 5          # Edge-Fog: 10ms ±5ms
#   ./tc-netem.sh eth0 50 10 0.1    # Fog-Cloud: 50ms ±10ms, 0.1% loss

IFACE="${1:-eth0}"
LATENCY_MS="${2:-10}"
JITTER_MS="${3:-5}"
LOSS_PCT="${4:-0}"

if ! command -v tc &> /dev/null; then
    echo "ERROR: tc not found — install iproute2"
    exit 1
fi

# Limpa regras existentes
tc qdisc del dev "$IFACE" root 2>/dev/null || true

# Aplica netem
if [ "$LOSS_PCT" != "0" ]; then
    tc qdisc add dev "$IFACE" root netem delay "${LATENCY_MS}ms" "${JITTER_MS}ms" loss "${LOSS_PCT}%" || {
        echo "ERROR: failed to apply netem with loss"
        exit 2
    }
else
    tc qdisc add dev "$IFACE" root netem delay "${LATENCY_MS}ms" "${JITTER_MS}ms" || {
        echo "ERROR: failed to apply netem"
        exit 2
    }
fi

echo "OK: $IFACE latency=${LATENCY_MS}ms jitter=${JITTER_MS}ms loss=${LOSS_PCT}%"
