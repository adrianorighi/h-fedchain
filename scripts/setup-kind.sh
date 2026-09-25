#!/usr/bin/env bash
set -euo pipefail

# Install kind + kubectl to ~/.local/bin (official release binaries, arch
# aware) and create the h-fedchain kind cluster from deploy/kind-config.yaml
# if it does not exist yet. Idempotent: re-runs skip downloads/creation.

KIND_VERSION="v0.30.0"
KUBECTL_VERSION="v1.33.4"
CLUSTER_NAME="h-fedchain"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/deploy/kind-config.yaml"
BIN_DIR="${KIND_BIN_DIR:-$HOME/.local/bin}"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[setup-kind]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup-kind]${NC} $*"; }
die()  { echo -e "${RED}[setup-kind] ERROR:${NC} $*" >&2; exit 1; }

arch() {
    case "$(uname -m)" in
        x86_64)  echo amd64 ;;
        aarch64|arm64) echo arm64 ;;
        *) die "unsupported architecture: $(uname -m)" ;;
    esac
}

# download <url> <dest> — two attempts, then fail (per task rule).
download() {
    local url="$1" dest="$2" attempt
    for attempt in 1 2; do
        log "downloading $url (attempt $attempt/2)"
        if curl -fsSL --connect-timeout 15 --max-time 600 -o "${dest}.tmp" "$url"; then
            mv "${dest}.tmp" "$dest"
            return 0
        fi
        sleep 2
    done
    rm -f "${dest}.tmp"
    return 1
}

# existing_bin <name> <version-string> <check-args...> → prints the path of a
# binary whose check output contains <version-string>; 1 when absent/wrong.
existing_bin() {
    local name="$1" want="$2" cand
    shift 2
    for cand in "$BIN_DIR/$name" "$(command -v "$name" 2>/dev/null || true)"; do
        [ -n "$cand" ] && [ -x "$cand" ] || continue
        if "$cand" "$@" 2>/dev/null | grep -q "$want"; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

# Sets KIND_BIN / KUBECTL_BIN globals.
ensure_kind() {
    if KIND_BIN="$(existing_bin kind "$KIND_VERSION" version)"; then
        log "kind already present: $KIND_BIN ($KIND_VERSION)"
        return 0
    fi
    mkdir -p "$BIN_DIR"
    download \
        "https://github.com/kubernetes-sigs/kind/releases/download/${KIND_VERSION}/kind-linux-$(arch)" \
        "$BIN_DIR/kind" \
        || die "kind download failed after 2 attempts (network blocked?)"
    chmod +x "$BIN_DIR/kind"
    "$BIN_DIR/kind" version | grep -q "$KIND_VERSION" \
        || die "downloaded kind does not report $KIND_VERSION"
    KIND_BIN="$BIN_DIR/kind"
    log "installed kind to $KIND_BIN"
}

ensure_kubectl() {
    if KUBECTL_BIN="$(existing_bin kubectl "$KUBECTL_VERSION" version --client)"; then
        log "kubectl already present: $KUBECTL_BIN ($KUBECTL_VERSION)"
        return 0
    fi
    mkdir -p "$BIN_DIR"
    download \
        "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/$(arch)/kubectl" \
        "$BIN_DIR/kubectl" \
        || die "kubectl download failed after 2 attempts (network blocked?)"
    chmod +x "$BIN_DIR/kubectl"
    "$BIN_DIR/kubectl" version --client | grep -q "$KUBECTL_VERSION" \
        || die "downloaded kubectl does not report $KUBECTL_VERSION"
    KUBECTL_BIN="$BIN_DIR/kubectl"
    log "installed kubectl to $KUBECTL_BIN"
}

case ":$PATH:" in
    *":$BIN_DIR:"*) : ;;
    *) warn "$BIN_DIR is not on PATH — add: export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

ensure_kind
ensure_kubectl

[ -f "$CONFIG_FILE" ] || die "missing $CONFIG_FILE"

# --- cluster ---
if "$KIND_BIN" get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    log "kind cluster '$CLUSTER_NAME' already exists"
else
    log "creating kind cluster '$CLUSTER_NAME' from $CONFIG_FILE"
    "$KIND_BIN" create cluster --name "$CLUSTER_NAME" \
        --config "$CONFIG_FILE" --wait 120s \
        || die "kind cluster creation failed"
fi

log "verifying:"
"$KIND_BIN" get clusters
"$KUBECTL_BIN" cluster-info || die "kubectl cluster-info failed"
log "done"
