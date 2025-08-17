import pandas as pd
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

# === Config ===
DATA_PATH = "train.csv"
INDEX_PATH = "faiss_overview.index"
TEXT_LIST_PATH = "id_to_text.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

# === Load and clean overviews ===
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df[df["overview"].notna()]
df = df[df["overview"].str.strip() != ""]
overviews = df["overview"].drop_duplicates().tolist()
print(f"Loaded {len(overviews)} unique overviews.")

# === Embed with SentenceTransformer ===
print(f"Encoding with {EMBED_MODEL}...")
encoder = SentenceTransformer(EMBED_MODEL)
embeddings = encoder.encode(overviews, show_progress_bar=True)
embeddings = np.array(embeddings).astype(np.float32)

# === Build FAISS index ===
print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# === Step 4: Save index and overview list ===
print(f"Saving index to {INDEX_PATH}...")
faiss.write_index(index, INDEX_PATH)

print(f"Saving overview list to {TEXT_LIST_PATH}...")
with open(TEXT_LIST_PATH, "w") as f:
    json.dump(overviews, f)

print("Done! FAISS index and overview list saved.")
