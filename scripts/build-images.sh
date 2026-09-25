#!/usr/bin/env bash
set -euo pipefail

# Build the services image and side-load it into the kind cluster.
# Idempotent (docker layer cache + kind load re-check), prints timings.
#
#   docker build -f services/Dockerfile -t hfedchain-services:latest .
#   kind load docker-image hfedchain-services:latest --name h-fedchain

IMAGE="hfedchain-services:latest"
CLUSTER_NAME="h-fedchain"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${CYAN}[build-images]${NC} $*"; }
die()  { echo -e "${RED}[build-images] ERROR:${NC} $*" >&2; exit 1; }

KIND_BIN="${KIND:-$(command -v kind || echo "$HOME/.local/bin/kind")}"
[ -x "$KIND_BIN" ] || die "kind not found — run scripts/setup-kind.sh first"

# --- build ---
t0=$(date +%s)
log "docker build -f services/Dockerfile -t $IMAGE . (repo root)"
docker build -f "$PROJECT_DIR/services/Dockerfile" -t "$IMAGE" "$PROJECT_DIR"
t1=$(date +%s)
log "build OK in $((t1 - t0))s"

# --- load into kind ---
if "$KIND_BIN" get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    log "kind load docker-image $IMAGE --name $CLUSTER_NAME"
    "$KIND_BIN" load docker-image "$IMAGE" --name "$CLUSTER_NAME"
    t2=$(date +%s)
    log "load OK in $((t2 - t1))s (total $((t2 - t0))s)"
else
    warn_msg="cluster '$CLUSTER_NAME' not found — skipping kind load (run scripts/setup-kind.sh first)"
    echo -e "${YELLOW}[build-images] WARNING:${NC} $warn_msg" >&2
fi

log "done: $IMAGE"
