from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import ast
from typing import List, Dict, Any
import os
from functools import lru_cache

app = FastAPI(title="Tagline Generator API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OMDb API Key
OMDB_API_KEY = "a167b60b"

# Cache data loading functions
@lru_cache(maxsize=1)
def load_metadata():
    df = pd.read_csv("movies_metadata.csv")
    df = df[df["overview"].notna() & df["tagline"].notna()]

    def extract_genres(genres_str):
        try:
            genres_list = ast.literal_eval(genres_str)
            return ", ".join([g["name"] for g in genres_list]) if isinstance(genres_list, list) else "N/A"
        except:
            return "N/A"

    df["parsed_genres"] = df["genres"].apply(extract_genres)
    return df

@lru_cache(maxsize=1)
def load_baseline():
    return pd.read_csv("generated_vs_original_with_beam.csv")

@lru_cache(maxsize=1)
def load_rag_infer():
    return pd.read_csv("generated_vs_original_RAG_infer.csv")

@lru_cache(maxsize=1)
def load_genre_rag():
    return pd.read_csv("generated_vs_original_genre_RAG-final.csv")

@lru_cache(maxsize=1)
def load_genre_only():
    return pd.read_csv("generated_vs_original_genre-final.csv")

@lru_cache(maxsize=1)
def load_genre_rag_boosted():
    return pd.read_csv("generated_vs_original_genre_boosted_RAG-final.csv")

@lru_cache(maxsize=1)
def load_genre_boosted():
    return pd.read_csv("generated_vs_original_genre_boost-final.csv")

@lru_cache(maxsize=1)
def get_valid_titles():
    df_meta = load_metadata()
    df_base = load_baseline()
    df_rag_infer = load_rag_infer()
    df_genre_rag = load_genre_rag()
    df_genre_only = load_genre_only()
    df_genre_rag_boosted = load_genre_rag_boosted()
    df_genre_boosted = load_genre_boosted()

    valid_titles = set(df_meta["title"].str.lower()) \
        & set(df_base["Title"].str.lower()) \
        & set(df_rag_infer["Title"].str.lower()) \
        & set(df_genre_rag["Title"].str.lower()) \
        & set(df_genre_only["Title"].str.lower()) \
        & set(df_genre_rag_boosted["Title"].str.lower()) \
        & set(df_genre_boosted["Title"].str.lower())

    return sorted({title for title in df_meta["title"] if title.lower() in valid_titles})

def is_valid_url(url):
    try:
        r = requests.head(url, timeout=5)
        return r.status_code == 200
    except:
        return False

def fetch_omdb_poster(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        res = requests.get(url, timeout=5).json()
        poster = res.get("Poster", "")
        if poster and poster != "N/A" and is_valid_url(poster):
            return poster
    except:
        pass
    return None

@app.get("/")
async def root():
    return {"message": "Tagline Generator API", "version": "1.0.0"}

@app.get("/movies", response_model=List[str])
async def get_movies():
    """Get list of all available movie titles"""
    return get_valid_titles()

@app.get("/movie/{title}")
async def get_movie_details(title: str):
    """Get movie details and generated taglines for a specific movie"""
    df_meta = load_metadata()
    df_base = load_baseline()
    df_rag_infer = load_rag_infer()
    df_genre_rag = load_genre_rag()
    df_genre_only = load_genre_only()
    df_genre_rag_boosted = load_genre_rag_boosted()
    df_genre_boosted = load_genre_boosted()

    # Find movie in metadata
    match = df_meta[df_meta["title"].str.lower() == title.strip().lower()]
    
    if match.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    row = match.iloc[0]
    movie_title = row["title"]
    overview = row["overview"]
    original_tagline = row["tagline"]
    poster_path = row.get("poster_path", "")
    backdrop_path = row.get("backdrop_path", "")
    genre = row.get("parsed_genres", "N/A")

    # Get poster URL
    poster_url = None
    tmdb_poster = f"https://image.tmdb.org/t/p/w500{poster_path}" if pd.notna(poster_path) else ""
    tmdb_backdrop = f"https://image.tmdb.org/t/p/w500{backdrop_path}" if pd.notna(backdrop_path) else ""
    
    if tmdb_poster and is_valid_url(tmdb_poster):
        poster_url = tmdb_poster
    elif tmdb_backdrop and is_valid_url(tmdb_backdrop):
        poster_url = tmdb_backdrop
    else:
        poster_url = fetch_omdb_poster(movie_title)

    # Get generated taglines
    def get_generated_tagline(df, title_col="Title"):
        match_row = df[df[title_col].str.lower() == movie_title.lower()]
        return match_row.iloc[0]["Generated"] if not match_row.empty else "Not found"

    baseline_tagline = get_generated_tagline(df_base)
    rag_infer_tagline = get_generated_tagline(df_rag_infer)
    genre_rag_tagline = get_generated_tagline(df_genre_rag)
    genre_only_tagline = get_generated_tagline(df_genre_only)
    genre_rag_boosted_tagline = get_generated_tagline(df_genre_rag_boosted)
    genre_boosted_tagline = get_generated_tagline(df_genre_boosted)

    return {
        "title": movie_title,
        "overview": overview,
        "original_tagline": original_tagline,
        "genres": genre.split(", ") if genre != "N/A" else [],
        "poster_url": poster_url,
        "generated_taglines": {
            "baseline": baseline_tagline,
            "rag_infer": rag_infer_tagline,
            "genre_only": genre_only_tagline,
            "genre_rag": genre_rag_tagline,
            "genre_rag_boosted": genre_rag_boosted_tagline,
            "genre_boosted": genre_boosted_tagline
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
