#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_viscom.sh --preset arp [--gpus 4090:2]
EOF
}

PRESET="arp"
GPUS="4090:2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="${2:-}"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$PRESET" ]]; then
  echo "Error: --preset is required." >&2
  usage
  exit 1
fi

case "$PRESET" in
  arp) CONFIG="classic_arp.yaml" ;;
  noarp) CONFIG="classic_noarp.yaml" ;;
  simple) CONFIG="simple_arp.yaml" ;;
  vanilla) CONFIG="classic_vanilla.yaml" ;;
  *) echo "Error: unknown preset '$PRESET'." >&2; usage; exit 1 ;;
esac

GPU_COUNT="${GPUS##*:}"
if [[ -z "$GPU_COUNT" || ! "$GPU_COUNT" =~ ^[0-9]+$ || "$GPU_COUNT" -lt 1 ]]; then
  echo "Error: --gpus must look like 6000:2 (count >= 1)." >&2
  exit 1
fi

DEVICES=""
for ((i=0; i<GPU_COUNT; i++)); do
  if [[ -n "$DEVICES" ]]; then
    DEVICES+=" "
  fi
  DEVICES+="cuda:$i"
done

COMMAND="whereis python && python ./bar-jepa/main.py --mode decoder --fname ./bar-jepa/configs/keypoint/$CONFIG --devices $DEVICES"
echo "Submitting preset '$PRESET' with config ./bar-jepa/configs/keypoint/$CONFIG and GPUs $GPUS"

submit "$COMMAND" \
  --custom dralois/ijepa-decoder:latest \
  --gpus "$GPUS" \
  --name ijepa_decoder \
  --max-time 1-0
