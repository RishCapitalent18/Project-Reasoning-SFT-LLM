#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:4
#SBATCH --partition=l40s_normal_q       # or a100_normal_q / a30_normal_q
#SBATCH --time=24:00:00
#SBATCH --account=ece_6514


# === Environment Setup ===
module load Miniconda3
module load CUDA/12.6.0
source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# === Model & Runtime Settings ===
NUM_GPUS=1
MODEL=/home/$USER/A-TML/LLaMA-Factory/saves/qwen25_3b_instruct/your_name1
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# === Output Directory ===
OUTPUT_DIR=/home/$USER/A-TML/LLaMA-Factory/eval_results
mkdir -p $OUTPUT_DIR

echo " Starting evaluation for your fine-tuned SFT model..."

# === 1️⃣ Math, Science, and Code Reasoning ===
lighteval vllm $MODEL_ARGS \
"lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0,extended|lcb:codegeneration|0|0" \
--save-details \
--output-dir $OUTPUT_DIR

# === 2️⃣ General Knowledge (MMLU-Redux-2) ===
lighteval vllm $MODEL_ARGS \
"lighteval|mmlu_redux_2|0|0" \
--output-dir $OUTPUT_DIR

echo " Evaluation complete! Results saved to $OUTPUT_DIR"
