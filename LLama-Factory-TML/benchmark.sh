#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=1:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=ece_6514
#SBATCH --gres=gpu:1

module load Miniconda3
module load CUDA/12.6.0
source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1
MODEL=Qwen/Qwen2.5-3B-Instruct
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=/home/apoorvachavali/A-TML/LLaMA-Factory/outputs_bench

lighteval vllm $MODEL_ARGS "lighteval|mmlu_redux_2|0|0" \
    --output-dir $OUTPUT_DIR \
    --no-save-details
