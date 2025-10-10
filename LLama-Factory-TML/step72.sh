#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_normal_q     # or a100_normal_q
#SBATCH --time=30:00:00
#SBATCH --account=ece_6514


module load Miniconda3
module load CUDA/12.6.0
source activate myenv2

# Set this environment variable so LLaMA-Factory uses torchrun/DeepSpeed
export FORCE_TORCHRUN=1

# Optional: set a local Triton cache path to avoid NFS warnings
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
mkdir -p $TRITON_CACHE_DIR

echo " Running training with sft_config2.yaml ... advanced data selection techniques applied."
# Run training
llamafactory-cli train sft_config2.yaml
