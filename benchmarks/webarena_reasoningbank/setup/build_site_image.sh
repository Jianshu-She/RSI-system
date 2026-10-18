#!/bin/bash
# Build the WebArena shopping site image for rootless enroot (no docker/sudo needed).
# Source: WebArena-Verified's shopping image (same data as the original WebArena shopping_final_0712: 104,368 products).
set -e
source "$(dirname "$0")/../env.sh" > /dev/null
mkdir -p "$(dirname "$WA_SITE_IMAGE")"
[ -f "$WA_SITE_IMAGE" ] && { echo "exists: $WA_SITE_IMAGE"; exit 0; }
enroot import -o "$WA_SITE_IMAGE" docker://am1n3e/webarena-verified-shopping:0.1.0
echo "image: $WA_SITE_IMAGE  (start a site: bash scripts/start_site.sh <name> <offset>)"
