#!/usr/bin/env bash
set -euo pipefail

# Deploy the H-FedChain cloud to the kind cluster and wire BOTH directions
# of the split topology (broker/fogs/edges in compose, cloud in kind):
#
#   kind pod -> compose broker : MQTT_BROKER in k8s/cloud-config is the
#       `kind` network gateway (host) by default; verified with a socket
#       probe from the cloud pod and, if unreachable, falls back to
#       attaching hfc-broker to the kind network and using that IP.
#   compose fogs -> kind cloud : cloud Service is NodePort 30052 (listens on
#       the kind node's addresses incl. its compose-network IP); the compose
#       network is connected to the kind node container so
#       CLOUD_ADDRESS=h-fedchain-control-plane:30052 resolves (set by the
#       deploy/docker-compose.e2e.yml override).
#
# Idempotent: safe to re-run (applies are declarative, connect is guarded).

CLUSTER_NAME="h-fedchain"
NODE_CONTAINER="h-fedchain-control-plane"
BROKER_CONTAINER="hfc-broker"
NS="h-fedchain"
PLACEHOLDER="mosquitto.local"
GRPC_NODEPORT=30052

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_DIR/k8s"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[deploy-cloud]${NC} $*"; }
warn() { echo -e "${YELLOW}[deploy-cloud]${NC} $*"; }
die()  { echo -e "${RED}[deploy-cloud] ERROR:${NC} $*" >&2; exit 1; }

KIND_BIN="${KIND:-$(command -v kind || echo "$HOME/.local/bin/kind")}"
KUBECTL_BIN="${KUBECTL:-$(command -v kubectl || echo "$HOME/.local/bin/kubectl")}"
[ -x "$KIND_BIN" ] || die "kind not found — run scripts/setup-kind.sh first"
[ -x "$KUBECTL_BIN" ] || die "kubectl not found — run scripts/setup-kind.sh first"

# --- preconditions: cluster node container up -------------------------------
docker inspect -f '{{.State.Running}}' "$NODE_CONTAINER" 2>/dev/null | grep -q true \
    || die "kind node '$NODE_CONTAINER' not running — run scripts/setup-kind.sh first"

# kind network gateway == host address reachable from cluster pods.
# The kind network may be dual-stack: pick the IPv4 gateway (the IPv6
# entry — e.g. fc00:f853:...::1 — is not routable from pods here).
GATEWAY="$(docker network inspect kind -f '{{range .IPAM.Config}}{{println .Gateway}}{{end}}' 2>/dev/null \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)"
[ -n "$GATEWAY" ] || die "no IPv4 gateway on docker network 'kind' — cluster missing (scripts/setup-kind.sh)"

broker_running() {
    docker inspect -f '{{.State.Running}}' "$BROKER_CONTAINER" 2>/dev/null | grep -q true
}

# MQTT_BROKER substitution: sed a temp copy of the configmap (explicit, no
# kustomize dependency).
apply_configmap() { # $1 = broker host for the pods
    local broker_host="$1" tmp
    tmp="$(mktemp)"
    sed "s/${PLACEHOLDER}/${broker_host}/" "$K8S_DIR/cloud-configmap.yaml" >"$tmp"
    "$KUBECTL_BIN" apply -f "$tmp"
    rm -f "$tmp"
    log "MQTT_BROKER -> ${broker_host} (configmap cloud-config applied)"
}

pod_mqtt_ok() { # probe broker reachability FROM the cloud pod
    "$KUBECTL_BIN" -n "$NS" exec deploy/cloud -- \
        python -c "import socket; s=socket.create_connection((\"$1\", 1883), 5); s.close(); print('ok')" \
        2>/dev/null | grep -q ok
}

# --- manifests ---------------------------------------------------------------
log "applying namespace + cloud manifests"
"$KUBECTL_BIN" apply -f "$K8S_DIR/namespace.yaml"
apply_configmap "$GATEWAY"
"$KUBECTL_BIN" apply -f "$K8S_DIR/cloud-pvc.yaml"
"$KUBECTL_BIN" apply -f "$K8S_DIR/cloud-service.yaml"
"$KUBECTL_BIN" apply -f "$K8S_DIR/cloud-deployment.yaml"

# The WORM PVC persists across e2e runs (kind hostPath), so rows from the
# previous run would stay — reset before the restart or e2e criteria could
# pass on stale data (and round 1 shows up as a duplicate). Best-effort: on
# first install there is no pod to exec into yet.
log "resetting WORM ledger (persisted PVC, previous run's rows)"
"$KUBECTL_BIN" -n "$NS" exec deploy/cloud -- \
    sh -c 'rm -f /var/lib/hfc/worm.db*' 2>/dev/null \
    || warn "no running cloud pod yet — WORM not reset (first install?)"

# ConfigMap env is baked into the pod at start: roll the cloud so every
# re-run picks up the freshly substituted MQTT_BROKER (and any new image
# side-loaded into kind). Idempotent — harmless when nothing changed.
log "restarting cloud to pick up configmap/image changes"
"$KUBECTL_BIN" -n "$NS" rollout restart deploy/cloud >/dev/null

log "waiting for rollout (deploy/cloud, 120s)"
if ! "$KUBECTL_BIN" -n "$NS" rollout status deploy/cloud --timeout=120s; then
    die "cloud rollout failed — hint: run scripts/build-images.sh (image must be side-loaded with 'kind load'), then 'kubectl -n $NS describe pod -l component=cloud'"
fi

# --- verify pod -> broker (gateway), with kind-network fallback ---------------
if broker_running; then
    if pod_mqtt_ok "$GATEWAY"; then
        log "MQTT reachable from cloud pod via gateway $GATEWAY:1883"
    else
        warn "gateway $GATEWAY:1883 unreachable from cloud pod — falling back: attach $BROKER_CONTAINER to the kind network"
        docker network connect kind "$BROKER_CONTAINER" 2>/dev/null || true
        BROKER_KIND_IP="$(docker inspect -f '{{with index .NetworkSettings.Networks "kind"}}{{.IPAddress}}{{end}}' "$BROKER_CONTAINER")"
        [ -n "$BROKER_KIND_IP" ] || die "could not get $BROKER_CONTAINER IP on the kind network"
        apply_configmap "$BROKER_KIND_IP"
        "$KUBECTL_BIN" -n "$NS" rollout restart deploy/cloud
        "$KUBECTL_BIN" -n "$NS" rollout status deploy/cloud --timeout=120s
        pod_mqtt_ok "$BROKER_KIND_IP" \
            || die "MQTT still unreachable from the cloud pod (tried $GATEWAY and $BROKER_KIND_IP)"
        log "MQTT reachable from cloud pod via broker kind-network IP $BROKER_KIND_IP"
    fi
else
    warn "$BROKER_CONTAINER not running — skipping pod->broker probe (start the compose broker first for a full check)"
fi

# --- wire compose network -> kind node (fogs reach NodePort $GRPC_NODEPORT) ---
if broker_running; then
    COMPOSE_NET="$(docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{if ne $k "kind"}}{{println $k}}{{end}}{{end}}' "$BROKER_CONTAINER" | head -1)"
else
    COMPOSE_NET="$(docker network ls --format '{{.Name}}' | grep -E '_hfc-net$' | head -1 || true)"
fi
if [ -z "${COMPOSE_NET:-}" ]; then
    warn "compose network not found — run 'docker compose -f docker-compose.distributed.yml up -d broker' then re-run this script so the fogs can resolve $NODE_CONTAINER"
else
    if docker network inspect "$COMPOSE_NET" -f '{{.Containers}}' | grep -q "$NODE_CONTAINER"; then
        log "kind node already connected to compose network '$COMPOSE_NET'"
    else
        docker network connect "$COMPOSE_NET" "$NODE_CONTAINER" \
            || die "docker network connect $COMPOSE_NET $NODE_CONTAINER failed"
        log "connected compose network '$COMPOSE_NET' to kind node '$NODE_CONTAINER'"
    fi
fi

# --- status ------------------------------------------------------------------
log "status:"
"$KUBECTL_BIN" -n "$NS" get pods -o wide
"$KUBECTL_BIN" -n "$NS" get svc cloud
echo
log "cloud gRPC for compose fogs:  CLOUD_ADDRESS=${NODE_CONTAINER}:${GRPC_NODEPORT}"
log "host gRPC (debug, optional):  127.0.0.1:${GRPC_NODEPORT}"
log "done"
