#!/bin/bash
# Activate Dockerfile.xarta (secrets-baked variant), build, then restore clean Dockerfile.
# Usage: bash deploy.xarta.sh [--build] [--up]
# Designed for Dockerfile.xarta which is gitignored and contains any xarta-specific secrets.
set -e

STACK_NAME="pockettts-openai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPLOY_ENV="$SCRIPT_DIR/.xarta/deploy.env"
[[ -f "$DEPLOY_ENV" ]] && source "$DEPLOY_ENV"

: "${XARTA_LXC_ID:?Missing XARTA_LXC_ID}"
LXC_ID="$XARTA_LXC_ID"
TARGET_DIR="/opt/stacks/${STACK_NAME}"

DO_BUILD=false; DO_UP=false
for arg in "$@"; do
    case $arg in
        --build) DO_BUILD=true ;;
        --up)    DO_UP=true; DO_BUILD=true ;;
    esac
done

if [[ ! -f "$SCRIPT_DIR/Dockerfile.xarta" ]]; then
    echo "No Dockerfile.xarta found — nothing to activate."
    echo "Create Dockerfile.xarta with any secrets or xarta-specific overrides."
    exit 1
fi

echo "Pushing Dockerfile.xarta as Dockerfile on LXC $LXC_ID..."
pct exec $LXC_ID -- mkdir -p "$TARGET_DIR"
pct push $LXC_ID "$SCRIPT_DIR/Dockerfile.xarta" "$TARGET_DIR/Dockerfile"

$DO_BUILD && { echo "Building with Dockerfile.xarta..."; pct exec $LXC_ID -- bash -c "cd $TARGET_DIR && docker compose build"; }
$DO_UP    && { echo "Starting stack..."; pct exec $LXC_ID -- bash -c "cd $TARGET_DIR && docker compose up -d"; }

echo ""
echo "Done. Dockerfile.xarta is now active in $TARGET_DIR."
echo "Note: The deployed Dockerfile on LXC $LXC_ID now contains xarta secrets."
echo "      Run 'bash deploy.sh --build' to revert to the clean Dockerfile."
