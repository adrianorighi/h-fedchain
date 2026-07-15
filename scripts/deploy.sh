#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/config"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.distributed.yml"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'

usage() {
    cat <<EOF
H-FedChain Distributed Deploy

Usage:
  ./scripts/deploy.sh [command]

Commands:
  setup         Install dependencies (python + docker)
  build         Build Docker images for Fog and Cloud
  push          Push images to registry
  deploy        Deploy to VMs via SSH
  status        Check service status on all nodes
  logs          Tail logs from a service
  stop          Stop all services
  clean         Remove containers and images

Interactive mode: run without arguments for menu.
EOF
    exit 0
}

ensure_config() {
    if [ ! -f "$CONFIG_DIR/topology.json" ]; then
        echo -e "${YELLOW}[!] config/topology.json not found. Creating default...${NC}"
        mkdir -p "$CONFIG_DIR"
        cat > "$CONFIG_DIR/topology.json" <<'EOF'
{
  "fog_nodes": [
    {"id": "fog1", "host": "fog1.local", "port": 50051}
  ],
  "cloud": {"host": "cloud.local", "port": 50052},
  "mqtt_broker": {"host": "mqtt.local", "port": 1883},
  "edge_devices": ["d0", "d1", "d2"],
  "f": 1,
  "n": 3
}
EOF
        echo "  Edit $CONFIG_DIR/topology.json with your VM addresses."
    fi
}

cmd_setup() {
    echo -e "${CYAN}[setup] Installing Python dependencies...${NC}"
    cd "$PROJECT_DIR"
    python3 -m venv .venv 2>/dev/null || true
    source .venv/bin/activate
    pip install -e ".[dev,distributed]" 2>&1 | tail -3

    if command -v docker &>/dev/null; then
        echo -e "${GREEN}[setup] Docker found${NC}"
    else
        echo -e "${YELLOW}[setup] Install Docker: https://docs.docker.com/engine/install/${NC}"
    fi
    echo -e "${GREEN}[setup] Done${NC}"
}

cmd_build() {
    ensure_config
    echo -e "${CYAN}[build] Building Docker images...${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.distributed.yml build
    echo -e "${GREEN}[build] Done${NC}"
}

cmd_push() {
    TAG="${1:-latest}"
    ensure_config
    echo -e "${CYAN}[push] Pushing images...${NC}"
    docker tag h-fedchain-fog "registry.example.com/h-fedchain/fog:$TAG"
    docker tag h-fedchain-cloud "registry.example.com/h-fedchain/cloud:$TAG"
    docker push "registry.example.com/h-fedchain/fog:$TAG"
    docker push "registry.example.com/h-fedchain/cloud:$TAG"
    echo -e "${GREEN}[push] Done${NC}"
}

cmd_deploy() {
    ensure_config
    FOG_HOSTS=$(python3 -c "import json; [print(n['host']) for n in json.load(open('$CONFIG_DIR/topology.json'))['fog_nodes']]" 2>/dev/null || echo "")
    CLOUD_HOST=$(python3 -c "import json; print(json.load(open('$CONFIG_DIR/topology.json'))['cloud']['host'])" 2>/dev/null || echo "")

    for host in $FOG_HOSTS $CLOUD_HOST; do
        [ -z "$host" ] && continue
        echo -e "${CYAN}[deploy] Copying to $host...${NC}"
        rsync -avz --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
            "$PROJECT_DIR/" "ubuntu@$host:/opt/h-fedchain/"
        echo -e "${CYAN}[deploy] Starting services on $host...${NC}"
        ssh "ubuntu@$host" "cd /opt/h-fedchain && docker compose -f docker-compose.distributed.yml up -d" || true
    done
    echo -e "${GREEN}[deploy] Done${NC}"
}

cmd_status() {
    ensure_config
    FOG_HOSTS=$(python3 -c "import json; [print(n['host']) for n in json.load(open('$CONFIG_DIR/topology.json'))['fog_nodes']]" 2>/dev/null)
    CLOUD_HOST=$(python3 -c "import json; print(json.load(open('$CONFIG_DIR/topology.json'))['cloud']['host'])" 2>/dev/null)

    for host in $FOG_HOSTS; do
        echo -e "${CYAN}[status] Fog $host:${NC}"
        ssh "ubuntu@$host" "curl -s http://localhost:8000/health 2>/dev/null || echo 'unreachable'" || echo "  SSH failed"
    done
    if [ -n "$CLOUD_HOST" ]; then
        echo -e "${CYAN}[status] Cloud $CLOUD_HOST:${NC}"
        ssh "ubuntu@$CLOUD_HOST" "curl -s http://localhost:8001/health 2>/dev/null || echo 'unreachable'" || echo "  SSH failed"
    fi
}

cmd_logs() {
    ensure_config
    SERVICE="${1:-fog}"
    HOST="${2:-}"
    if [ -z "$HOST" ]; then
        HOST=$(python3 -c "
import json; data = json.load(open('$CONFIG_DIR/topology.json'))
if '$SERVICE' == 'cloud':
    print(data['cloud']['host'])
else:
    print(data['fog_nodes'][0]['host'])
" 2>/dev/null || echo "localhost")
    fi
    echo -e "${CYAN}[logs] Tailing $SERVICE on $HOST...${NC}"
    ssh "ubuntu@$HOST" "cd /opt/h-fedchain && docker compose -f docker-compose.distributed.yml logs -f $SERVICE"
}

cmd_stop() {
    ensure_config
    echo -e "${YELLOW}[stop] Stopping all services...${NC}"
    python3 -c "
import json
data = json.load(open('$CONFIG_DIR/topology.json'))
for n in data.get('fog_nodes', []):
    print(n['host'])
print(data.get('cloud', {}).get('host', ''))
" 2>/dev/null | while read host; do
        [ -z "$host" ] && continue
        ssh "ubuntu@$host" "cd /opt/h-fedchain && docker compose -f docker-compose.distributed.yml down" 2>/dev/null || true
    done
    echo -e "${GREEN}[stop] Done${NC}"
}

cmd_clean() {
    echo -e "${YELLOW}[clean] Removing containers and images...${NC}"
    docker compose -f "$COMPOSE_FILE" down --rmi all 2>/dev/null || true
    echo -e "${GREEN}[clean] Done${NC}"
}

# === Menu ===
menu() {
    while true; do
        echo ""
        echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║         H-FedChain Distributed Deploy        ║${NC}"
        echo -e "${CYAN}╠══════════════════════════════════════════════╣${NC}"
        echo -e "${CYAN}║${NC}  1. Setup environment (venv + deps)         ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  2. Build Docker images                     ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  3. Deploy to VMs (SSH + rsync)            ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  4. Check service status                    ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  5. Tail logs (fog/cloud)                  ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  6. Stop all services                       ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  7. Run tests (pytest)                      ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  8. Run simulation (local, single-process)  ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  9. Clean containers/images                 ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  0. Exit                                    ${CYAN}║${NC}"
        echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
        echo ""
        read -rp "  Choose [0-9]: " choice
        case "$choice" in
            1) cmd_setup ;;
            2) cmd_build ;;
            3) cmd_deploy ;;
            4) cmd_status ;;
            5)
                read -rp "  Service (fog/cloud): " svc
                cmd_logs "$svc"
                ;;
            6) cmd_stop ;;
            7)
                cd "$PROJECT_DIR"
                source .venv/bin/activate 2>/dev/null || true
                pytest -v
                ;;
            8)
                cd "$PROJECT_DIR"
                source .venv/bin/activate 2>/dev/null || true
                python -m experiments.scenario_1_nominal
                ;;
            9) cmd_clean ;;
            0) echo "Bye."; exit 0 ;;
            *) echo -e "${RED}Invalid option${NC}" ;;
        esac
        echo ""
        read -rp "Press Enter to continue..."
    done
}

mkdir -p "$CONFIG_DIR/results"
[ $# -eq 0 ] && menu

case "${1:-help}" in
    setup) cmd_setup ;;
    build) cmd_build ;;
    push) cmd_push "${2:-latest}" ;;
    deploy) cmd_deploy ;;
    status) cmd_status ;;
    logs) cmd_logs "${2:-fog}" "${3:-}" ;;
    stop) cmd_stop ;;
    clean) cmd_clean ;;
    help|--help|-h) usage ;;
    *) echo -e "${RED}Unknown command: $1${NC}"; usage ;;
esac
