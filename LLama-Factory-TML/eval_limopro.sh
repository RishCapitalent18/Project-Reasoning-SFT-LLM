#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=30:00:00
#SBATCH --partition=l40s_normal_q
#SBATCH --account=ece_6514
#SBATCH --gres=gpu:4
#SBATCH --job-name=eval_limopro
#SBATCH --output=logs/eval_limopro_%j.out

module load Miniconda3
module load CUDA/12.6.0

source activate myenv

export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=4

# Your LIMOPro model path
MODEL=/home/apoorvachavali/A-TML/LLaMA-Factory/qwen_limopro_output/checkpoint-705

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=/home/apoorvachavali/A-TML/eval_results/limopro

mkdir -p $OUTPUT_DIR

echo "Evaluating LIMOPro model..."
lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0,extended|lcb:codegeneration|0|0,lighteval|mmlu_redux_2|0|0" \
    --save-details \
    --output-dir $OUTPUT_DIR

echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"