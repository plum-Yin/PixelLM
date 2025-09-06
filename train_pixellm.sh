#!/bin/bash


# ======== SLURM Job Configuration ========
#SBATCH --job-name=pixellm_train
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --open-mode=truncate
#SBATCH --output=/data/zyin418/mllm/PixelLM/output/output.log
#SBATCH --error=/data/zyin418/mllm/PixelLM/output/error.log

# ======== Job Execution Steps ========

# Navigate to the working directory where your code and virtual environment are located
cd /data/zyin418/mllm/PixelLM

# Activate the Python virtual environment (adjust if it's named differently)
# 加载 conda 初始化脚本
source /data/zyin418/miniconda3/etc/profile.d/conda.sh
conda activate pixellm


# Run your Python script or other commands
# export PIXELLM_DEBUG=1

deepspeed --num_gpus 1 --master_port=24999 train_ds.py \
    --dataset reason_seg \
    --sample_rates 1 \
    --dataset_dir ../dataset \
    --preprocessor_config configs/preprocessor_224.json \
    --vision_pretrained sam_vit_h_4b8939.pth \
    --version "liuhaotian/llava-v1.6-vicuna-7b" \
    --conv_type "llava_llama_2" \
    --epochs 10 \
    --steps_per_epoch 5 \
    --batch_size 32 \
