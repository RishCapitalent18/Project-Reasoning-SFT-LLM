#!/bin/bash
# Evaluate the random 15k SFT model using lighteval+vLLM
# Outputs saved to submission/results/raw

#-- SLURM Job Directives (edit as needed) --#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_normal_q
#SBATCH --time=8:00:00
#SBATCH --account=ece_6514

set -euo pipefail

# Environment (ARC)
module load Miniconda3
module load CUDA/12.6.0
source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn
PROJECT_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
NUM_GPUS=1

# Update this path to your actual trained model directory if different
MODEL="${PROJECT_ROOT}/saves/qwen25_3b_instruct/your_name1"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.90,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR="${PROJECT_ROOT}/submission/results/raw"
mkdir -p "$OUTPUT_DIR"

echo "Starting evaluation for SFT (random) model at $MODEL ..."
# Math/science/code reasoning
lighteval vllm $MODEL_ARGS \
  "lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0,extended|lcb:codegeneration|0|0" \
  --save-details \
  --output-dir "$OUTPUT_DIR"

# MMLU-Redux-2 general benchmark
lighteval vllm $MODEL_ARGS \
  "lighteval|mmlu_redux_2|0|0" \
  --output-dir "$OUTPUT_DIR"

echo "Evaluation complete. Results saved to: $OUTPUT_DIR"
