#!/bin/bash
# Push templates/ and voices/ to the Docker host LXC without rebuilding the image.
# Volume-mounted files take effect immediately — no restart required.
# Usage: bash deploy-ui.sh
set -e

STACK_NAME="pockettts-openai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Infrastructure config -----------------------------------------------
DEPLOY_ENV="$SCRIPT_DIR/.xarta/deploy.env"
[[ -f "$DEPLOY_ENV" ]] && source "$DEPLOY_ENV"

: "${XARTA_LXC_ID:?Missing XARTA_LXC_ID — copy deploy.env.example to .xarta/deploy.env and fill in}"
LXC_ID="$XARTA_LXC_ID"
TARGET_TEMPLATES="/opt/stacks/${STACK_NAME}/templates"
# -------------------------------------------------------------------------

SRC_TEMPLATES="$SCRIPT_DIR/templates"

pct exec $LXC_ID -- mkdir -p "$TARGET_TEMPLATES"
pct push $LXC_ID "$SRC_TEMPLATES/index.html" "$TARGET_TEMPLATES/index.html"

echo "UI deployed to $TARGET_TEMPLATES. Refresh browser."
