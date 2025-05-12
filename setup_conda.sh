#!/bin/bash
# === HPC MODULE SETUP ===
echo "🔧 Loading Gridware environment and Python module..."
source /opt/flight/etc/setup.sh
flight env activate gridware
module add gnu
module load python/3.9.7

export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128

ls ~/miniconda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tag_gen_env
which python
python --version

# === CHECK AND INSTALL MINICONDA ===
MINICONDA_DIR="$HOME/miniconda3" 
if ! command -v conda &> /dev/null; then
    echo "📦 Conda not found. Installing Miniconda..."

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm miniconda.sh

    # Enable conda for this shell
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"  # 👈 Add this
    conda init bash
    echo "✅ Miniconda installed at $MINICONDA_DIR"
else
    echo "✅ Conda already available."
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
fi



# === CREATE CONDA ENVIRONMENT ===
ENV_NAME="tag_gen_env"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "✅ Conda environment '$ENV_NAME' already exists — skipping creation."
else
    echo "📦 Creating conda environment '$ENV_NAME' with Python 3.9..."
    conda create -n "$ENV_NAME" python=3.9 -y
fi

# === ACTIVATE CONDA ENV ===
echo "🚀 Activating conda environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# === INSTALL REQUIREMENTS ===
echo "📦 Installing packages from requirements.txt..."
pip install --proxy http://hpc-proxy00.city.ac.uk:3128 -r requirements.txt

# === INSTALL PYTORCH (CUDA) ===
echo "🔥 Installing CUDA 11.8-compatible PyTorch..."
pip install --proxy http://hpc-proxy00.city.ac.uk:3128 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# === W&B Auto Login ===
echo "🔐 Logging into Weights & Biases..."
export WANDB_API_KEY="f1b1dcb5ebf893b856630d4481b7d7cd05101b45"  
wandb login --relogin "$WANDB_API_KEY"

# === CONFIRM ===
echo "✅ Setup complete!"


source ~/miniconda3/etc/profile.d/conda.sh
conda activate tag_gen_env
which python
python --version

