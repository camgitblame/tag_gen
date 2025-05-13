#!/bin/bash
#SBATCH --job-name=tagline-inference          # Job name
#SBATCH --partition=gengpu                    # GPU partition
#SBATCH --nodes=1                             # 1 node
#SBATCH --ntasks-per-node=1                   # 1 task per node
#SBATCH --cpus-per-task=4                     # 4 CPU cores
#SBATCH --mem=16GB                            # RAM
#SBATCH --time=01:00:00                       # Max runtime
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --output=results/%x_%j.out            # Stdout log
#SBATCH --error=results/%x_%j.err             # Stderr log

# === ENV SETUP ===
echo "🔧 Initializing environment..."
source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu

export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128
export TORCH_HOME=/mnt/data/public/torch

# Activate conda and env
echo "🐍 Activating Conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tag_gen_env

# Print device info
echo "🔍 CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "⚠️ Could not run nvidia-smi"
python -c "import torch; print('✅ CUDA Available:', torch.cuda.is_available())"


# 🌐 Login to WandB
export WANDB_API_KEY=f1b1dcb5ebf893b856630d4481b7d7cd05101b45
echo "🔐 Logging into WandB..."
wandb login $WANDB_API_KEY --relogin

# === Run inference ===
echo "🚀 Running inference..."
python inference_upgraded.py
