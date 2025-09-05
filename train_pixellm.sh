#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=pixellm_train

#SBATCH --time=00:30:00
# [REQUIRED] Wall time limit in HH:MM:SS (adjust to your expected runtime)

#SBATCH --open-mode=append

#SBATCH --output=/data/zyin418/mllm/PixelLM/output/output.log

#SBATCH --error=/data/zyin418/mllm/PixelLM/output/error.log

#SBATCH --gres=gpu:1

# ======== Job Execution Steps ========

# Navigate to the working directory where your code and virtual environment are located

cd /data/zyin418/mllm/PixelLM

# Activate the Python virtual environment (adjust if it's named differently)
# 加载 conda 初始化脚本
source /data/zyin418/miniconda3/etc/profile.d/conda.sh
conda activate pixellm


# Run your Python script or other commands
echo "Starting training job..."
deepspeed --num_gpus 1 train_ds.py \
    --dataset reason_seg \
    --sample_rates 1 \
    --dataset_dir ../dataset \
    --preprocessor_config configs/preprocessor_448.json \
    --vision_pretrained sam_vit_h_4b8939.pth
    # --model_name meta-llama/Llama-3.1-8B-Instruct \
    