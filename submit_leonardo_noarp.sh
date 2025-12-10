#!/bin/bash
#SBATCH --job-name=train-noarp          # job name
#SBATCH --time=12:00:00                 # time limit
#SBATCH --nodes=1                       # 1 node
#SBATCH --gres=gpu:4                    # 4 GPUs
#SBATCH --mem=220GB                     # 55GB per GPU x 4 GPUs = 220GB total
#SBATCH --ntasks-per-node=4             # 4 tasks per node (one per GPU)
#SBATCH --cpus-per-task=8               # 10 CPUs per task
#SBATCH --output=slurm_logs/logs-%j.out # standard output file
#SBATCH --error=slurm_logs/logs-%j.err  # standard error file
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --qos=boost_qos_bprod           # QoS for up to 1 days

# Load any required modules
module load profile/deeplrn
module load cineca-ai/4.1.1

# Debug
nvidia-smi

# Navigate to working directory
cd $WORK/ijepa-bars/ijepa-encoder

# Execute your script
python main_finetune.py --fname configs/charts_vith14_noarp.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3