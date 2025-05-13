#!/bin/bash
#SBATCH --job-name=tagline-gpu-train         # Job name
#SBATCH --partition=gengpu                   # GPU partition
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --nodes=1                            # Run on 1 node
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=4                    # 4 CPU cores
#SBATCH --mem=24G                            # RAM
#SBATCH --time=01:00:00                      # Max wall time
#SBATCH -o results/%x_%j.out                 # Stdout
#SBATCH -e results/%x_%j.err                 # Stderr

# === LOAD MODULES ===
source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu

# === ACTIVATE ENV ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tag_gen_env

# === PROXY CONFIG (if needed) ===
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128

# === VERIFY GPU ===
echo "CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('✅ CUDA Available:', torch.cuda.is_available())"


# 🌐 Login to WandB
export WANDB_API_KEY=f1b1dcb5ebf893b856630d4481b7d7cd05101b45
echo "🔐 Logging into WandB..."
wandb login $WANDB_API_KEY --relogin

export WANDB_PROJECT=tag_gen

# === TRAIN ===
python scripts/run_clm.py \
  --model_name_or_path gpt2 \
  --tokenizer_name tokenizer \
  --train_file train_augmented.txt \
  --do_train \
  --do_eval \
  --output_dir gpt2-RAG-output \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --block_size 512 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --logging_steps 50 \
  --report_to wandb \
  --run_name "RAG-training" \
  --load_best_model_at_end \
  --save_total_limit 2 \
  --fp16


