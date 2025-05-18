The requirements.txt file was re-built as we added new modules to the code to make it easier to revert back to a safe version.
The requirements_genre_rag.txt should contain all the necessary packages for all of the models.
The training command to train each model is located in the corresponding run.sh scripts.
With the training completed and the gpt2-output folder(s) created, the corresponding inference.py file can be run with "python inference.py".
Each inference file contains a field for MODEL_DIR and EVAL_FILE. 
MODEL_DIR should be gpt2-output for normal models, gpt2-output-genre, or gpt2-output-genre-boosted based on the model being used. These folders are created after training.
The genre inference files are currently configured with the boosted model.
EVAL_FILE should be eval_genre.csv for genre models and eval.csv for normal models, as the genre csvs has an extra column.
Each inference file generates a "generated_vs..." csv that shows a side by side comparison of the generated taglines with the original taglines.


# Movie Tagline Generator

This project fine-tunes variants of GPT-2 to generate movie taglines based on metadata such as plot overviews and genres. It includes baseline generation, genre conditioning, and retrieval-augmented generation (RAG). It supports training, inference, evaluation with ROUGE and BERTScore, and includes a Streamlit demo for exploration.

---

[View our Streamlit App](https://taggen.streamlit.app/)

[View our W&B Report](https://api.wandb.ai/links/camgitblame-city-university-of-london/xuucaogs)

---

## Project Structure

```
.
├── run.sh                 # SLURM script for training
├── test.sh                # SLURM script for inference
├── setup_conda.sh         # Conda env setup with CUDA and proxy
├── inference.py           # Generates taglines and computes metrics
├── preprocess_taglines.py # Cleans and formats mdata
├── create_eval_csv.py     # Creates eval.csv for inference testing
├── movies_metadata.csv    # Raw dataset
├── requirements.txt       # Python dependencies
```

---

## Environment Setup

### 1. HPC Environment Setup

If you're using  Hyperion:

```bash
source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu
module load python/3.9.7
```

### 2. Create and Activate Conda Environment

Run the setup script (handles proxy, CUDA, pip install, and login to W&B):

```bash
source setup_conda.sh
```

Alternatively, do it manually:

```bash
# One-time setup
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
conda create -n tag_gen_env python=3.9 -y
conda activate tag_gen_env
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Training the Model

Use the provided script to train baseline model:

```bash
sbatch run.sh
```

This launches `scripts/run_clm.py` to fine-tune GPT-2 on `train.txt`. 


## 🔍 Inference & Evaluation

To generate taglines and compute evaluation metrics:

```bash
sbatch test.sh
```

This script:

- Loads `gpt2-output` as the model
- Uses `eval.csv` for testing. 
- Runs `inference.py` to generate taglines and log metrics:
  - **ROUGE-1 / ROUGE-L** for lexical overlap
  - **BERTScore F1** for semantic similarity

Results are logged to [Weights & Biases](https://wandb.ai/).


## Streamlit App 

A companion Streamlit app can be used to interactively explore generated tagline.

Run the app locally

```bash
streamlit run app.py
```
We deployed at:

https://taggen.streamlit.app/


## Dataset

If using this work, cite the dataset:

```
@dataset{rounakbanik_2017,
  author       = {Rounak Banik},
  title        = {The Movies Dataset},
  year         = 2017,
  url          = {https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset}
}
```
