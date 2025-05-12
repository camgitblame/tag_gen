#!/bin/bash

# === HPC MODULE SETUP ===
echo "🔧 Loading Gridware environment and Python module..."
source /opt/flight/etc/setup.sh
flight env activate gridware
module add gnu
module load python/3.9.7



export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128


# === W&B Auto Login ===
echo "🔐 Logging into Weights & Biases..."
export WANDB_API_KEY="f1b1dcb5ebf893b856630d4481b7d7cd05101b45"  
wandb login --relogin "$WANDB_API_KEY"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tag_gen_env
which python
python --version




