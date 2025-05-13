## Install Miniconda and Set Up Your HPC Python Environment

### 📍 Prerequisites:

- You are logged into **Hyperion**.
- You’re in your project directory (e.g., `tag_gen/`).
- You have a valid `requirements.txt` file in that directory.

Before starting, enter a login shell:

```bash
bash --login
```

First Time Set Up

```bash
source ./setup_conda.sh 
```

Reactivate Conda for each session

```bash
source ./activate_conda.sh
```

---

### 1. **Load the HPC Gridware Environment**

```bash
# Set up system modules and environment
source /opt/flight/etc/setup.sh
flight env activate gridware

# Load required modules
module load libs/nvidia-cuda/11.2.0/bin
module load gnu
module load python/3.9.7
```

---

### 2. **Set Up Proxy (Required for Network Access)**

```bash
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128
```

---

### 3. **Install Miniconda (if not already installed)**

```bash
# Define the install path
MINICONDA_DIR="$HOME/miniconda3"

# Check if conda is already installed
if ! command -v conda &> /dev/null; then
    echo "📦 Conda not found. Installing Miniconda..."

    # Download and install
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm miniconda.sh

    echo "✅ Miniconda installed at $MINICONDA_DIR"
fi

```

---

### 4. **Initialize and Source Conda**

```bash
# Enable conda in the current shell
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash

```

---

### 5. **Create Your Conda Environment**

```bash
# Define environment name
ENV_NAME="tag_gen_env"

# Create environment if it doesn't exist
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "📦 Creating conda environment '$ENV_NAME' with Python 3.9..."
    conda create -n "$ENV_NAME" python=3.9 -y
fi

```

---

### 6. **Activate the Conda Environment**

[]()

```bash
echo "🚀 Activating conda environment..."
conda activate tag_gen_env

```

---

### 7. **Install Python Dependencies**

```bash
# Install packages from requirements.txt
pip install --proxy "$https_proxy" -r requirements.txt

```

---

### 8. **Install PyTorch with CUDA 11.8 Support**

```bash
pip install --proxy "$https_proxy" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

---

### 9. **Verify Installation**

```bash
which python
python --version

```

Expected output should look like:

```bash
/users/YOUR_USERNAME/miniconda3/envs/tag_gen_env/bin/python
Python 3.9.x

```

---

### 10. **Log Into Weights & Biases**
