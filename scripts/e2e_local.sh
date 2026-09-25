#!/usr/bin/env bash
set -euo pipefail

# H-FedChain local end-to-end (T14).
#
# Topology under test:
#   broker + {fog1,fog1b}/c1 + {fog2,fog2b}/c2 + edge/c1 + edge/c2
#                                                       -> docker compose
#                                                       (deploy/docker-compose.e2e.yml
#                                                        override: ROUNDS=3,
#                                                        CLOUD_ADDRESS=kind node)
#   cloud (inter-cluster aggregator, N_EXPECTED_CLUSTERS=2) -> kind cluster
#                                                       (Service NodePort 30052)
#
# Sequence: setup-kind -> build+load image -> clean compose slate -> broker up
#           -> deploy-cloud (MQTT gateway substitution + network connect)
#           -> fogs+edges up -> wait (<= WAIT_CAP_S) -> PASS/FAIL criteria.
#
# Usage:
#   scripts/e2e_local.sh                  full run
#   scripts/e2e_local.sh --skip-build     reuse existing images
#   scripts/e2e_local.sh --down           compose down -v (teardown)
#   scripts/e2e_local.sh --down --delete-kind    ...also delete the kind cluster
#
# Success-criteria note (log evidence): per-round flushed prints on every
# component form the evidence chain (AuditLogger/QC_COMMIT is simulator-only):
#   fog  "[node] round R: cluster=... role=leader"     (leader elected in-cluster)
#         + "... committed round R (ledger height=H)"  (QC commit -> ledger append)
#         + "... submitted cluster=... round=R to cloud OK"  (post-append only)
#   cloud "... received cluster=cX round=R (k/2)"      -> BOTH clusters present
#         + "[cloud] round R aggregated ... broadcast=yes"
#   edge "[edge] round N/3 applied"                    <- models/global (retained)
# WORM rows in the cloud pod prove the aggregation side directly.

ROUNDS_E2E=3
WAIT_CAP_S=900          # edge-wait cap (10-15 min per task budget)
POLL_S=10

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.distributed.yml"
E2E_OVERRIDE="$PROJECT_DIR/deploy/docker-compose.e2e.yml"
NS="h-fedchain"
CLUSTER_NAME="h-fedchain"

SKIP_BUILD=0
DO_DOWN=0
DELETE_KIND=0

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[e2e]${NC} $*"; }
warn() { echo -e "${YELLOW}[e2e]${NC} $*"; }
die()  { echo -e "${RED}[e2e] ERROR:${NC} $*" >&2; exit 1; }

usage() {
    cat <<EOF
H-FedChain local e2e (compose broker/fogs/edges + kind cloud)

Usage: scripts/e2e_local.sh [--skip-build] [--down] [--delete-kind]

  --skip-build     skip docker build + kind load (reuse existing image)
  --down           tear the compose stack down (down -v --remove-orphans)
  --delete-kind    also delete the kind cluster (implies --down's scope)
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)   SKIP_BUILD=1 ;;
        --down)         DO_DOWN=1 ;;
        --delete-kind)  DELETE_KIND=1 ;;
        -h|--help)      usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
    shift
done

compose() { docker compose -f "$COMPOSE_FILE" -f "$E2E_OVERRIDE" "$@"; }
KUBECTL_BIN="${KUBECTL:-$(command -v kubectl || echo "$HOME/.local/bin/kubectl")}"
KIND_BIN="${KIND:-$(command -v kind || echo "$HOME/.local/bin/kind")}"

# deploy-cloud.sh attaches the kind node to the compose network so the fogs
# can reach NodePort 30052. compose's `down` HANGS (observed: indefinite,
# compose v5.5.1) when removing a network that still has that foreign
# container endpoint — detach it first. Idempotent; re-attached by the next
# deploy-cloud.sh run.
detach_kind_node() {
    local net
    for net in $(docker network ls --format '{{.Name}}' | grep -E '_hfc-net$' || true); do
        docker network disconnect "$net" h-fedchain-control-plane 2>/dev/null || true
    done
}

# ---------------------------------------------------------------- teardown --
dump_logs() {
    echo
    echo "════════════════════ E2E failure dump (last 50 lines each) ════════════════════"
    echo "── compose ps ──"
    compose ps -a 2>&1 || true
    local svc
    for svc in broker fog1 fog1b fog2 fog2b edge edge-c2; do
        echo "── compose logs: $svc ──"
        compose logs --no-log-prefix --tail=50 "$svc" 2>&1 || true
    done
    echo "── kubectl get pods ──"
    "$KUBECTL_BIN" -n "$NS" get pods -o wide 2>&1 || true
    echo "── cloud pod logs (tail 50) ──"
    local cp
    cp=$(cloud_pod)
    "$KUBECTL_BIN" -n "$NS" logs "${cp:-deploy/cloud}" --tail=50 --timestamps 2>&1 || true
    echo "── cloud deployment status ──"
    "$KUBECTL_BIN" -n "$NS" describe deploy/cloud 2>&1 | tail -30 || true
    echo "════════════════════ end failure dump ════════════════════"
}

DUMP_ON_FAIL=1
on_exit() {
    local rc=$?
    if [ "$DUMP_ON_FAIL" -eq 1 ] && [ "$rc" -ne 0 ]; then
        dump_logs
        echo -e "${RED}[e2e] FAILED (exit $rc)${NC}"
    fi
}
trap on_exit EXIT

if [ "$DO_DOWN" -eq 1 ] || [ "$DELETE_KIND" -eq 1 ]; then
    DUMP_ON_FAIL=0
    log "teardown: compose down -v --remove-orphans"
    detach_kind_node
    timeout 180 docker compose -f "$COMPOSE_FILE" -f "$E2E_OVERRIDE" \
        down -v --remove-orphans --timeout 20 2>&1 \
        || warn "compose down failed or timed out (check: docker ps -a)"
    if [ "$DELETE_KIND" -eq 1 ]; then
        log "deleting kind cluster '$CLUSTER_NAME'"
        "$KIND_BIN" delete cluster --name "$CLUSTER_NAME" 2>&1 || true
    fi
    log "teardown done"
    exit 0
fi

# ------------------------------------------------------------------ helpers --
edge_state() { # "<status>/<exitcode>" (missing container -> missing/0)
    docker inspect -f '{{.State.Status}}/{{.State.ExitCode}}' "$1" 2>/dev/null || echo "missing/0"
}

# Newest cloud pod (rollout may leave the old pod Terminating for ~30s;
# `kubectl logs deploy/cloud` is ambiguous with 2 pods and picks the old one,
# whose log is from the PREVIOUS run). Sort by creationTimestamp.
cloud_pod() {
    "$KUBECTL_BIN" -n "$NS" get pods -l component=cloud \
        -o jsonpath='{range .items[*]}{.metadata.creationTimestamp}{" "}{.metadata.name}{"\n"}{end}' \
        2>/dev/null | sort | tail -1 | cut -d' ' -f2
}

# Fixed-string checks against captured log text. Pure bash `case` with a
# quoted needle: literal match, no pipes/grep/pipefail pitfalls.
present() { # <needle> <haystack>
    case "$2" in *"$1"*) return 0 ;; *) return 1 ;; esac
}

CRIT_PASS=0
CRIT_FAIL=0
crit_pass() { CRIT_PASS=$((CRIT_PASS + 1)); echo -e "  ${GREEN}PASS${NC}  $*"; }
crit_fail() { CRIT_FAIL=$((CRIT_FAIL + 1)); echo -e "  ${RED}FAIL${NC}  $*"; }
crit_present() { # label needle haystack
    if present "$2" "$3"; then crit_pass "$1"; else crit_fail "$1 — missing: $2"; fi
}
crit_absent() { # label needle haystack
    if present "$2" "$3"; then crit_fail "$1 — found: $2"; else crit_pass "$1"; fi
}

# =================================================================== steps ==

log "step 1/7 — kind cluster (setup-kind.sh)"
"$SCRIPT_DIR/setup-kind.sh"

log "step 2/7 — build + kind-load image"
if [ "$SKIP_BUILD" -eq 1 ]; then
    log "--skip-build: reusing existing images"
else
    "$SCRIPT_DIR/build-images.sh"
fi

log "step 3/7 — clean compose slate (down -v)"
detach_kind_node
timeout 180 docker compose -f "$COMPOSE_FILE" -f "$E2E_OVERRIDE" \
    down -v --remove-orphans --timeout 20 \
    || die "compose down failed/timed out"

log "step 4/7 — broker up + port check"
compose up -d --no-deps broker
BROKER_OK=0
for _ in $(seq 1 30); do
    if python3 -c 'import socket; socket.create_connection(("127.0.0.1", 1883), 1).close()' 2>/dev/null; then
        BROKER_OK=1
        break
    fi
    sleep 1
done
[ "$BROKER_OK" -eq 1 ] || die "mosquitto did not open host port 1883 within 30s"
log "broker up (1883)"

log "step 5/7 — deploy cloud to kind + wire networks (deploy-cloud.sh)"
"$SCRIPT_DIR/deploy-cloud.sh"

log "step 6/7 — fogs + edges up (ROUNDS=$ROUNDS_E2E via e2e override)"
compose up -d --no-deps --build fog1 fog1b fog2 fog2b edge edge-c2
compose ps

log "step 7/7 — waiting for both edges to finish round $ROUNDS_E2E (cap ${WAIT_CAP_S}s)"
wait_edges() {
    local deadline=$(( $(date +%s) + WAIT_CAP_S ))
    local s1 s2 cloud_bad=0 remaining
    while :; do
        remaining=$(( deadline - $(date +%s) ))
        if [ "$remaining" -le 0 ]; then
            log "wait cap reached (${WAIT_CAP_S}s)"
            return 1
        fi
        s1=$(edge_state hfc-edge-c1)
        s2=$(edge_state hfc-edge-c2)
        if [ "$s1" = "exited/0" ] && [ "$s2" = "exited/0" ]; then
            log "both edges exited cleanly"
            return 0
        fi
        # dead edge: exited non-zero — confirm it is not just the instant
        # between crash and the on-failure restart before giving up.
        case "$s1$s2" in
            *exited/[1-9]*)
                sleep 5
                s1=$(edge_state hfc-edge-c1); s2=$(edge_state hfc-edge-c2)
                case "$s1$s2" in
                    *exited/[1-9]*)
                        log "edge failed permanently (c1=$s1 c2=$s2)"
                        return 1 ;;
                esac
                ;;
        esac
        # cloud pod must stay ready
        local ready
        ready=$("$KUBECTL_BIN" -n "$NS" get deploy cloud \
                    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
        if [ "${ready:-0}" -lt 1 ] 2>/dev/null; then
            cloud_bad=$((cloud_bad + 1))
            if [ "$cloud_bad" -ge 2 ]; then
                log "cloud deployment lost readiness"
                return 1
            fi
        else
            cloud_bad=0
        fi
        log "waiting… edge c1=$s1 c2=$s2 (${remaining}s left)"
        sleep "$POLL_S"
    done
}

if ! wait_edges; then
    warn "edges did not complete successfully — running criteria anyway"
    WAIT_FAILED=1
else
    WAIT_FAILED=0
fi

# ================================================================ criteria ==
echo
echo "════════════════════ Success criteria ════════════════════"

edge_c1_logs=$(compose logs --no-log-prefix edge 2>/dev/null || true)
edge_c2_logs=$(compose logs --no-log-prefix edge-c2 2>/dev/null || true)
fog1_logs=$(compose logs --no-log-prefix fog1 2>/dev/null || true)
fog1b_logs=$(compose logs --no-log-prefix fog1b 2>/dev/null || true)
fog2_logs=$(compose logs --no-log-prefix fog2 2>/dev/null || true)
fog2b_logs=$(compose logs --no-log-prefix fog2b 2>/dev/null || true)
cloud_logs=$("$KUBECTL_BIN" -n "$NS" logs "$(cloud_pod)" --tail=10000 2>/dev/null || true)

# 1. edges completed round ROUNDS_E2E and summarized it, clean exit
crit_present "edge c1: applied final round $ROUNDS_E2E" \
    "[edge] round $ROUNDS_E2E/$ROUNDS_E2E applied" "$edge_c1_logs"
crit_present "edge c1: summary rounds_completed=$ROUNDS_E2E (exit 0)" \
    "rounds_completed=$ROUNDS_E2E" "$edge_c1_logs"
crit_present "edge c2: applied final round $ROUNDS_E2E" \
    "[edge] round $ROUNDS_E2E/$ROUNDS_E2E applied" "$edge_c2_logs"
crit_present "edge c2: summary rounds_completed=$ROUNDS_E2E (exit 0)" \
    "rounds_completed=$ROUNDS_E2E" "$edge_c2_logs"
if [ "$(edge_state hfc-edge-c1)" = "exited/0" ] && [ "$(edge_state hfc-edge-c2)" = "exited/0" ]; then
    crit_pass "edge containers: both exited with code 0"
else
    crit_fail "edge containers: c1=$(edge_state hfc-edge-c1) c2=$(edge_state hfc-edge-c2)"
fi

# 2. every round reached the edges through the cloud broadcast
for r in 1 2 "$ROUNDS_E2E"; do
    crit_present "edge c1: round $r broadcast applied" \
        "[edge] round $r/$ROUNDS_E2E applied" "$edge_c1_logs"
    crit_present "edge c2: round $r broadcast applied" \
        "[edge] round $r/$ROUNDS_E2E applied" "$edge_c2_logs"
done

# 3. cloud pod: gRPC actually listening (in-pod TCP probe — the startup
#    print is block-buffered without a TTY, so probe instead of grepping it),
#    config sanity, per-round aggregate+broadcast lines, no failure markers
grpc_probe=$("$KUBECTL_BIN" -n "$NS" exec "$(cloud_pod)" -- \
    python -c 'import socket; s=socket.create_connection(("127.0.0.1", 50052), 3); s.close(); print("grpc-ok")' \
    2>/dev/null || echo "grpc-fail")
if present "grpc-ok" "$grpc_probe"; then
    crit_pass "cloud: gRPC listening on :50052 (in-pod probe)"
else
    crit_fail "cloud: gRPC listening on :50052 (in-pod probe) — got: $grpc_probe"
fi
crit_present "cloud: config expected_clusters=2" \
    "[cloud] config: expected_clusters=2" "$cloud_logs"
for r in 1 2 "$ROUNDS_E2E"; do
    crit_present "cloud: round $r aggregated + broadcast" \
        "[cloud] round $r aggregated" "$cloud_logs"
done
# both cluster outputs must land for EVERY round (the pairing that was
# deadlocked by cross-cluster committees before Option B)
for r in 1 2 "$ROUNDS_E2E"; do
    crit_present "cloud: received c1 output for round $r" \
        "received cluster=c1 round=$r" "$cloud_logs"
    crit_present "cloud: received c2 output for round $r" \
        "received cluster=c2 round=$r" "$cloud_logs"
done
crit_absent "cloud: no MQTT connect failure" \
    "MQTT connection failed" "$cloud_logs"
crit_absent "cloud: no undelivered broadcast" \
    "not delivered" "$cloud_logs"
crit_absent "cloud: no traceback" \
    "Traceback" "$cloud_logs"

# 4. WORM ledger rows for rounds 1..R
worm=$("$KUBECTL_BIN" -n "$NS" exec "$(cloud_pod)" -- \
    python -c 'import sqlite3; c=sqlite3.connect("/var/lib/hfc/worm.db"); print("worm_rounds=" + ",".join(str(r[0]) for r in c.execute("SELECT round FROM global_outputs ORDER BY round")))' \
    2>/dev/null || echo "worm_query_failed")
worm_ok=1
for r in 1 2 "$ROUNDS_E2E"; do
    case ",${worm#worm_rounds=}," in
        *",$r,"*) : ;;
        *) worm_ok=0 ;;
    esac
done
if [ "$worm_ok" -eq 1 ] && present "worm_rounds=" "$worm"; then
    crit_pass "cloud WORM ledger has rounds 1..$ROUNDS_E2E ($worm)"
else
    crit_fail "cloud WORM ledger missing rounds 1..$ROUNDS_E2E ($worm)"
fi

# 5. fog committee operation (cluster-local, 2 members each): startup,
#    in-cluster leader election, QC commit (ledger append), gRPC submission —
#    plus absence of failure markers
for fog in fog1 fog1b fog2 fog2b; do
    case "$fog" in
        fog1)  flog="$fog1_logs";  fnode="fog1" ;;
        fog1b) flog="$fog1b_logs"; fnode="fog1b" ;;
        fog2)  flog="$fog2_logs";  fnode="fog2" ;;
        fog2b) flog="$fog2b_logs"; fnode="fog2b" ;;
    esac
    crit_present "$fog: service started" \
        "[$fnode] Fog service running" "$flog"
    crit_present "$fog: elected leader for a round" \
        "role=leader" "$flog"
    crit_present "$fog: QC commit appended to ledger" \
        "committed round" "$flog"
    crit_present "$fog: submitted cluster output over gRPC" \
        "submitted cluster=" "$flog"
    crit_absent "$fog: no cloud-submission failure" \
        "Cloud submission failed" "$flog"
    crit_absent "$fog: no quorum failure (=> no view-change rounds)" \
        "Quorum not reached" "$flog"
    crit_absent "$fog: no traceback" \
        "Traceback" "$flog"
done

# 6. broker retained broadcast for the final round (direct proof of cloud -> broker)
retained_file=$(mktemp)
docker exec hfc-broker mosquitto_sub -t models/global -C 1 -W 5 \
    >"$retained_file" 2>/dev/null || true
retained_round=$(python3 -c "import pickle,sys; print(pickle.load(open(sys.argv[1],'rb'))['round'])" "$retained_file" 2>/dev/null || echo "none")
rm -f "$retained_file"
if [ "$retained_round" = "$ROUNDS_E2E" ]; then
    crit_pass "broker: retained models/global is round $retained_round"
else
    crit_fail "broker: retained models/global round=$retained_round (expected $ROUNDS_E2E)"
fi

# =================================================================== result ==
echo
echo "════════════════════ Summary ════════════════════"
echo -e "  passed: ${GREEN}${CRIT_PASS}${NC}   failed: ${RED}${CRIT_FAIL}${NC}   wait_failed: ${WAIT_FAILED}"
if [ "$CRIT_FAIL" -eq 0 ] && [ "$WAIT_FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}E2E PASS${NC}"
    echo "  (stack left running — tear down with: scripts/e2e_local.sh --down)"
    exit 0
fi
echo -e "  ${RED}${BOLD}E2E FAIL${NC}"
exit 1
