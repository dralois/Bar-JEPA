#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_viscom.sh --mode decoder --preset arp [--gpus 4090:2]
  scripts/submit_viscom.sh --mode decoder --preset arp-heavypos [--gpus 4090:2]
  scripts/submit_viscom.sh --mode decoder --preset arp-pos1 [--gpus 4090:2]
  scripts/submit_viscom.sh --mode finetune --preset arp [--gpus 4090:2]
EOF
}

MODE="decoder"
PRESET="arp"
GPUS="4090:2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --preset) PRESET="${2:-}"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Error: --mode is required." >&2
  usage
  exit 1
fi

if [[ "$MODE" != "decoder" && "$MODE" != "finetune" ]]; then
  echo "Error: --mode must be 'decoder' or 'finetune'." >&2
  usage
  exit 1
fi

if [[ -z "$PRESET" ]]; then
  echo "Error: --preset is required." >&2
  usage
  exit 1
fi

if [[ "$MODE" == "decoder" ]]; then
  case "$PRESET" in
    arp) CONFIG="classic_arp.yaml" ;;
    aux-only) CONFIG="classic_aux_only.yaml" ;;
    hm-only) CONFIG="classic_hm_only.yaml" ;;
    noarp) CONFIG="classic_noarp.yaml" ;;
    simple) CONFIG="simple_arp.yaml" ;;
    vanilla) CONFIG="classic_vanilla.yaml" ;;
    *) echo "Error: unknown preset '$PRESET' for decoder." >&2; usage; exit 1 ;;
  esac
  CONFIG_DIR="keypoint"
else
  case "$PRESET" in
    arp) CONFIG="vith14_arp.yaml" ;;
    noarp) CONFIG="vith14_noarp.yaml" ;;
    *) echo "Error: unknown preset '$PRESET' for finetune." >&2; usage; exit 1 ;;
  esac
  CONFIG_DIR="charts"
fi

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

COMMAND="whereis python && python ./bar-jepa/main.py --mode $MODE --fname ./bar-jepa/configs/$CONFIG_DIR/$CONFIG --devices $DEVICES"
echo "Submitting mode '$MODE' preset '$PRESET' with config ./bar-jepa/configs/$CONFIG_DIR/$CONFIG and GPUs $GPUS"

submit "$COMMAND" \
  --custom dralois/ijepa-decoder:latest \
  --gpus "$GPUS" \
  --name ijepa_decoder \
  --max-time 1-0
