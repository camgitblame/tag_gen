#!/bin/bash
# === HPC MODULE SETUP ===
echo "🔧 Loading Gridware environment and Python module..."
source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu
module load python/3.9.7

# === PROXY CONFIG ===
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128



# === Conda Environment Activation ===
echo "🐍 Activating conda environment 'tag_gen_env'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tag_gen_env



# === Python Check ===
echo "✅ Environment ready!"
which python
python --version

# === W&B Auto Login ===
export WANDB_API_KEY=f1b1dcb5ebf893b856630d4481b7d7cd05101b45
echo "🔐 Logging into WandB..."
wandb login $WANDB_API_KEY --relogin