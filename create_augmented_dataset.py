import pandas as pd
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Parameters
DATA_PATH = "movies_metadata.csv"
OUTPUT_PATH = "train_augmented.txt"
# Number of retrieved examples
K = 2  

# Load and clean data
df = pd.read_csv(DATA_PATH)
df = df[df["overview"].notna() & df["tagline"].notna()]
df = df[df["overview"].str.strip() != ""]
df = df[df["tagline"].str.strip() != ""]
df = df.drop_duplicates(subset=["overview", "tagline"])

overviews = df["overview"].tolist()
taglines = df["tagline"].tolist()

# Step 2: Embed overviews
print("🔍 Encoding overviews...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(overviews, show_progress_bar=True)

# Step 3: Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Generate augmented prompts
print("🧠 Retrieving and formatting training examples...")
lines = []
for i in tqdm(range(len(overviews))):
    query = embeddings[i].reshape(1, -1)
    _, I = index.search(query, K + 1)  # +1 to skip the query itself

    # Filter out the current index
    retrieved = [overviews[j] for j in I[0] if j != i][:K]

    retrieved_block = "\n".join([f"Retrieved Overview {j+1}: {txt}" for j, txt in enumerate(retrieved)])
    prompt = f"{retrieved_block}\nOriginal Overview: {overviews[i]}\nTagline: {taglines[i]}"
    lines.append(prompt)

# Save to text file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line.strip() + "\n\n") 

print(f"✅ Saved {len(lines)} augmented prompts to {OUTPUT_PATH}")
