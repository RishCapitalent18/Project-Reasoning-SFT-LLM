#!/bin/bash
# Evaluate the LIMOPro 15k SFT model using lighteval+vLLM
# Outputs saved to submission/results/raw

#-- SLURM Job Directives (edit as needed) --#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=30:00:00
#SBATCH --partition=l40s_normal_q
#SBATCH --account=ece_6514
#SBATCH --gres=gpu:4
#SBATCH --job-name=eval_limopro

set -euo pipefail

# Environment (ARC)
module load Miniconda3
module load CUDA/12.6.0
source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn
PROJECT_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
NUM_GPUS=4

# Update this path to your actual LIMOPro checkpoint directory if different
MODEL="${PROJECT_ROOT}/saves/qwen25_3b_instruct/limopro_run/checkpoint-705"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR="${PROJECT_ROOT}/submission/results/raw"
mkdir -p "$OUTPUT_DIR"

echo "Evaluating LIMOPro model at $MODEL ..."
lighteval vllm $MODEL_ARGS \
  "lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0,extended|lcb:codegeneration|0|0,lighteval|mmlu_redux_2|0|0" \
  --save-details \
  --output-dir "$OUTPUT_DIR"

echo "Evaluation complete. Results saved to: $OUTPUT_DIR"
