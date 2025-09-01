from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import ast
import re
from typing import List, Dict, Any
import os
from functools import lru_cache

app = FastAPI(title="Tagline Generator API", version="1.0.0")

# Startup event to warm up critical caches


@app.on_event("startup")
async def startup_event():
    """Preload critical data to improve first request performance"""
    print("Starting up Tagline Generator API...")
    print("Preloading movie metadata...")
    load_metadata()  # This will cache the metadata
    print("API ready to serve requests!")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://tagline-r4u8sod5s-cams-projects-03a5c6f6.vercel.app",
        "https://tagline-pf2bbhu37-cams-projects-03a5c6f6.vercel.app",
        "https://tagline-gmcpnqvve-cams-projects-03a5c6f6.vercel.app",
        "https://tagline-474x9zlus-cams-projects-03a5c6f6.vercel.app",
        "https://tagline-gen.vercel.app",  # Main domain
        "*",  # Allow all origins to debug
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OMDb API Key
OMDB_API_KEY = "a167b60b"


# Cache data loading functions with optimizations
@lru_cache(maxsize=1)
def load_metadata():
    # Load only necessary columns to reduce memory usage
    usecols = ["title", "overview", "tagline", "genres",
               "poster_path", "release_date", "vote_average"]
    df = pd.read_csv("movies_metadata.csv", usecols=usecols, dtype={
        "title": "string",
        "overview": "string",
        "tagline": "string",
        "genres": "string",
        "poster_path": "string",
        "release_date": "string"
    }, low_memory=False)

    # Filter out null values early
    df = df.dropna(subset=["overview", "tagline"])
    df = df[(df["overview"].str.strip() != "") &
            (df["tagline"].str.strip() != "")]

    # Optimized genre parsing using regex instead of ast.literal_eval
    def extract_genres_fast(genres_str):
        if pd.isna(genres_str) or genres_str.strip() == "":
            return "N/A"
        try:
            # Use regex to extract genre names more efficiently than ast.literal_eval
            genre_names = re.findall(r"'name': '([^']+)'", genres_str)
            return ", ".join(genre_names) if genre_names else "N/A"
        except:
            return "N/A"

    df["parsed_genres"] = df["genres"].apply(extract_genres_fast)
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
    """
    Optimized computation of valid titles.
    Only loads datasets when needed and uses efficient set operations.
    """
    df_meta = load_metadata()

    # Load other datasets on demand
    df_base = load_baseline()
    df_rag_infer = load_rag_infer()
    df_genre_rag = load_genre_rag()
    df_genre_only = load_genre_only()
    df_genre_rag_boosted = load_genre_rag_boosted()
    df_genre_boosted = load_genre_boosted()

    # Pre-compute lowercase title sets for faster intersection
    meta_titles_lower = set(df_meta["title"].str.lower())
    base_titles_lower = set(df_base["Title"].str.lower())
    rag_infer_titles_lower = set(df_rag_infer["Title"].str.lower())
    genre_rag_titles_lower = set(df_genre_rag["Title"].str.lower())
    genre_only_titles_lower = set(df_genre_only["Title"].str.lower())
    genre_rag_boosted_titles_lower = set(
        df_genre_rag_boosted["Title"].str.lower())
    genre_boosted_titles_lower = set(df_genre_boosted["Title"].str.lower())

    # Efficient set intersection
    valid_titles_lower = (
        meta_titles_lower
        & base_titles_lower
        & rag_infer_titles_lower
        & genre_rag_titles_lower
        & genre_only_titles_lower
        & genre_rag_boosted_titles_lower
        & genre_boosted_titles_lower
    )

    # Return sorted list with original case
    return sorted({title for title in df_meta["title"] if title.lower() in valid_titles_lower})


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
    """Get list of all available movie titles with lazy loading"""
    try:
        return get_valid_titles()
    except Exception as e:
        print(f"Error loading movies: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to load movies list")


@app.get("/movies/count")
async def get_movies_count():
    """Get count of available movies (faster than loading full list)"""
    try:
        df_meta = load_metadata()
        return {"count": len(df_meta)}
    except Exception as e:
        print(f"Error getting movie count: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get movie count")


@app.get("/movie/{title}")
async def get_movie_details(title: str):
    """Get movie details and generated taglines for a specific movie"""
    try:
        # Load metadata first (should be cached from startup)
        df_meta = load_metadata()

        # Find movie in metadata
        match = df_meta[df_meta["title"].str.lower() == title.strip().lower()]

        if match.empty:
            raise HTTPException(status_code=404, detail="Movie not found")

        row = match.iloc[0]
        movie_title = row["title"]
        overview = row["overview"]
        original_tagline = row["tagline"]
        poster_path = row.get("poster_path", "")
        genre = row.get("parsed_genres", "N/A")

        # Get poster URL
        poster_url = None
        tmdb_poster = (
            f"https://image.tmdb.org/t/p/w500{poster_path}" if pd.notna(
                poster_path) else ""
        )

        if tmdb_poster and is_valid_url(tmdb_poster):
            poster_url = tmdb_poster
        else:
            poster_url = fetch_omdb_poster(movie_title)

        # Load generation datasets only when needed for this specific movie
        df_base = load_baseline()
        df_rag_infer = load_rag_infer()
        df_genre_rag = load_genre_rag()
        df_genre_only = load_genre_only()
        df_genre_rag_boosted = load_genre_rag_boosted()
        df_genre_boosted = load_genre_boosted()

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
                "genre_boosted": genre_boosted_tagline,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting movie details for '{title}': {e}")
        raise HTTPException(
            status_code=500, detail="Failed to load movie details")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
