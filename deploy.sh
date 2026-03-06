#!/bin/bash
# Deploy pockettts-openai to a Docker host via pct.
# Usage: bash deploy.sh [--dry-run] [--build] [--up]
#   --dry-run   Show what would happen without doing it
#   --build     Build Docker image after pushing files
#   --up        Start the stack after building (implies --build)
set -e

STACK_NAME="pockettts-openai"
TARGET_DIR="/opt/stacks/${STACK_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Infrastructure config -----------------------------------------------
# Source deploy environment (gitignored — never committed to public repos).
# Copy deploy.env.example → .xarta/deploy.env and fill in your values.
DEPLOY_ENV="$SCRIPT_DIR/.xarta/deploy.env"
[[ -f "$DEPLOY_ENV" ]] && source "$DEPLOY_ENV"

: "${XARTA_LXC_ID:?Missing XARTA_LXC_ID — copy deploy.env.example to .xarta/deploy.env and fill in}"
: "${XARTA_HOST_IP:?Missing XARTA_HOST_IP — see deploy.env.example}"
LXC_ID="$XARTA_LXC_ID"
PORT="${XARTA_HOST_PORT:-8884}"
# -------------------------------------------------------------------------

DRY_RUN=false; DO_BUILD=false; DO_UP=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --build)   DO_BUILD=true ;;
        --up)      DO_UP=true; DO_BUILD=true ;;
    esac
done

if $DRY_RUN; then
    echo "[dry-run] Would tar $SCRIPT_DIR → $TARGET_DIR on LXC $LXC_ID"
    exit 0
fi

pct status $LXC_ID | grep -q running || { echo "LXC $LXC_ID not running"; exit 1; }
pct exec $LXC_ID -- mkdir -p "$TARGET_DIR"

echo "Packaging and pushing repo..."
tar -czf /tmp/${STACK_NAME}.tar.gz \
    --exclude='.git' \
    --exclude='logs' \
    --exclude='outputs' \
    --exclude='hf_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.xarta' \
    --exclude='Dockerfile.xarta' \
    -C "$SCRIPT_DIR" .
pct push $LXC_ID /tmp/${STACK_NAME}.tar.gz /tmp/${STACK_NAME}.tar.gz
pct exec $LXC_ID -- bash -c "tar -xzf /tmp/${STACK_NAME}.tar.gz -C $TARGET_DIR && rm /tmp/${STACK_NAME}.tar.gz"
echo "  -> pushed to $TARGET_DIR"

$DO_BUILD && { echo "Building image..."; pct exec $LXC_ID -- bash -c "cd $TARGET_DIR && docker compose build"; }
$DO_UP    && { echo "Starting stack..."; pct exec $LXC_ID -- bash -c "cd $TARGET_DIR && docker compose up -d"; }

echo ""
echo "Done."
$DO_UP && echo "  Health: curl http://${XARTA_HOST_IP}:${PORT}/health"
$DO_UP && echo "  UI:     http://${XARTA_HOST_IP}:${PORT}/"
$DO_UP && echo "  Voices: curl http://${XARTA_HOST_IP}:${PORT}/v1/voices"
$DO_UP && echo "  Dockge: http://${XARTA_HOST_IP}:5001"
