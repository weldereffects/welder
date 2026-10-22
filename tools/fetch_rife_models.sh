#!/usr/bin/env bash
# Download the RIFE models used by the RIFE node into models/rife.
#
#   tools/fetch_rife_models.sh [dest]      (default: models/rife)
#
# The models are ncnn conversions of Practical-RIFE (github.com/hzwer/Practical-RIFE,
# MIT) from github.com/TNTwise/rife-ncnn-vulkan (MIT). Each model is a folder
# holding flownet.param and flownet.bin.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$ROOT/models/rife}
MODELS=(rife-v4.26 rife-v4.22-lite)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git clone -q --depth 1 --filter=blob:none --sparse https://github.com/TNTwise/rife-ncnn-vulkan "$TMP/tnt"
(cd "$TMP/tnt" && git sparse-checkout set $(printf 'models/%s ' "${MODELS[@]}") >/dev/null)
mkdir -p "$DEST"
for m in "${MODELS[@]}"; do
  rm -rf "$DEST/$m" && mkdir -p "$DEST/$m"
  cp "$TMP/tnt/models/$m/flownet.param" "$TMP/tnt/models/$m/flownet.bin" "$DEST/$m/"
  echo "  $m"
done
cp "$TMP/tnt/LICENSE" "$DEST/LICENSE-rife-ncnn-vulkan.txt" 2>/dev/null || true
cat > "$DEST/README.txt" <<'TXT'
RIFE models for Welder's RIFE node: ncnn conversions of Practical-RIFE
(https://github.com/hzwer/Practical-RIFE, MIT licence) from
https://github.com/TNTwise/rife-ncnn-vulkan (MIT licence).
Add another model by copying its folder (flownet.param + flownet.bin) here.
TXT
echo "RIFE models in $DEST"
