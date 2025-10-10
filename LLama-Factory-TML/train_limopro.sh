#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=30:00:00
#SBATCH --partition=l40s_normal_q
#SBATCH --account=ece_6514
#SBATCH --gres=gpu:4

module load Miniconda3
module load CUDA/12.6.0

source activate myenv2

# Create directories
mkdir -p /home/apoorvachavali/A-TML/LLaMA-Factory/saves/qwen25_3b_limopro
mkdir -p logs

echo "Starting LIMOPro training at $(date)"
echo "Dataset: acereason_limopro_15k (15,000 samples)"
echo "Output: /home/apoorvachavali/A-TML/LLaMA-Factory/saves/qwen25_3b_limopro"

# Start training
cd /home/apoorvachavali/A-TML/LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train sft_limopro.yaml

echo "Training completed at $(date)"