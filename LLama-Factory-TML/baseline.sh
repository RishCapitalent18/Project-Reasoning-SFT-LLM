#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=2         # Request 2 CPU cores
#SBATCH --time=1:00:00              # Set a 1-hour time limit
#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU

module load Miniconda3
module load CUDA/12.6.0

source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1
MODEL=Qwen/Qwen2.5-3B-Instruct
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=/home/apoorvachavali/A-TML/LLaMA-Factory/outputs_baseline

lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0,extended|lcb:codegeneration|0|0" \
    --save-details \
--output-dir $OUTPUT_DIR