#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_uc3.sh --mode test --arp
  scripts/submit_uc3.sh --mode train --partition gpu_h100 --no-arp

Options:
  --mode test|train           Required. Select test (20 min) or train (12 hours).
  --partition NAME            Required for train. e.g. gpu_h100 | gpu_a100_il | gpu_h100_il
  --arp                       Use ARP config (default)
  --no-arp                    Use no-ARP config
EOF
}

MODE=""
PARTITION=""
ARP_MODE="arp"

TIME=""
GPUS=""
CPUS_PER_TASK=""
MEM_PER_GPU="55G"
PROJECT_DIR="$HOME/Bar-JEPA"
WORKSPACE_PATH="/pfs/work9/workspace/scratch/ul_spm55-mydata-ssd"
DATASET_TGZ="data.tar"
DATASET_DIR=""
IMAGE="dralois/ijepa-decoder:latest"
LOG_DIR="slurm_logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --partition) PARTITION="${2:-}"; shift 2 ;;
    --arp) ARP_MODE="arp"; shift 1 ;;
    --no-arp) ARP_MODE="noarp"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Error: --mode is required." >&2
  usage
  exit 1
fi

if [[ "$MODE" != "test" && "$MODE" != "train" ]]; then
  echo "Error: --mode must be 'test' or 'train'." >&2
  usage
  exit 1
fi

if [[ "$MODE" == "test" ]]; then
  if [[ -n "$PARTITION" ]]; then
    case "$PARTITION" in
      dev_gpu_h100|gpu_a100_short) PARTITION="$PARTITION" ;;
      *) echo "Error: test mode supports dev_gpu_h100 or gpu_a100_short." >&2; exit 1 ;;
    esac
  else
    PARTITION="dev_gpu_h100"
  fi
  TIME="00:20:00"
  if [[ "$PARTITION" == "gpu_a100_short" ]]; then
    GPUS="1"
  else
    GPUS="4"
  fi
  CPUS_PER_TASK="10"
  MEM_PER_GPU="55G"
  JOB_NAME="arp-test"
  DATASET_DIR="data/test"
else
  if [[ -z "$PARTITION" ]]; then
    echo "Error: --partition is required for train mode." >&2
    usage
    exit 1
  fi
  TIME="12:00:00"
  GPUS="4"
  CPUS_PER_TASK="10"
  MEM_PER_GPU="55G"
  JOB_NAME="arp-train"
  DATASET_DIR="data/train"
fi

if [[ "$ARP_MODE" == "arp" ]]; then
  CONFIG_NAME="vith14_arp.yaml"
else
  CONFIG_NAME="vith14_noarp.yaml"
fi

DEVICES=""
for ((i=0; i<GPUS; i++)); do
  if [[ -n "$DEVICES" ]]; then
    DEVICES+=" "
  fi
  DEVICES+="cuda:$i"
done

mkdir -p "$LOG_DIR"

CONTAINER_MOUNTS="/etc/slurm/task_prolog:/etc/slurm/task_prolog,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15,$WORKSPACE_PATH:$WORKSPACE_PATH"
CONTAINER_MOUNTS_LINE="#SBATCH --container-mounts=$CONTAINER_MOUNTS"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --nodes=1
#SBATCH --gres=gpu:$GPUS
#SBATCH --ntasks-per-node=$GPUS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem-per-gpu=$MEM_PER_GPU
#SBATCH --output=$LOG_DIR/logs-%j.out
#SBATCH --error=$LOG_DIR/logs-%j.err
#SBATCH --partition=$PARTITION
#SBATCH --container-image=$IMAGE
#SBATCH --container-workdir=$PROJECT_DIR
#SBATCH --container-mount-home
$CONTAINER_MOUNTS_LINE

nvidia-smi

if [[ -n "\$TMPDIR" && -d "\$TMPDIR" ]]; then
  : # keep Slurm-provided TMPDIR if it exists
else
  export TMPDIR="/tmp/slurm_tmpdir_\${SLURM_JOB_ID}"
fi
mkdir -p "\$TMPDIR"

PROJECT_DIR=$PROJECT_DIR
WORKSPACE_PATH=$WORKSPACE_PATH
DATASET_TGZ=$DATASET_TGZ
DATASET_DIR=$DATASET_DIR

DATA_ROOT="\$TMPDIR/\$DATASET_DIR"
CONFIG_PATH="\$TMPDIR/${CONFIG_NAME%.yaml}-tmp.yaml"

if [[ -n "\$DATASET_TGZ" ]]; then
  SRC_TGZ="\$WORKSPACE_PATH/\$DATASET_TGZ"
  if [[ ! -f "\$SRC_TGZ" ]]; then
    echo "ERROR: dataset tarball not found: \$SRC_TGZ" >&2
    exit 1
  fi
  echo "Staging tarball \$SRC_TGZ to \$TMPDIR ..."
  tar -C "\$TMPDIR" -xf "\$SRC_TGZ"
fi

sed "s|^  root_path: .*|  root_path: \${DATA_ROOT}/|" "\$PROJECT_DIR/bar-jepa/configs/charts/$CONFIG_NAME" > "\$CONFIG_PATH"
echo "Using staged root_path: \${DATA_ROOT}/"

python \$PROJECT_DIR/bar-jepa/main.py \\
  --mode finetune \\
  --fname \$CONFIG_PATH \\
  --devices $DEVICES
EOF
