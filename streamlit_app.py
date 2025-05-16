import streamlit as st
import pandas as pd
import requests
import os
import ast

# === OMDb API Key ===
OMDB_API_KEY = "a167b60b"  

# === Load metadata and generations ===
@st.cache_data
def load_metadata():
    df = pd.read_csv("movies_metadata.csv")
    df = df[df["overview"].notna() & df["tagline"].notna()]

    # Convert stringified lists in the 'genres' column to actual lists
    def extract_genres(genres_str):
        try:
            genres_list = ast.literal_eval(genres_str)
            return ", ".join([g["name"] for g in genres_list]) if isinstance(genres_list, list) else "N/A"
        except:
            return "N/A"

    df["parsed_genres"] = df["genres"].apply(extract_genres)
    return df

@st.cache_data
def load_baseline():
    return pd.read_csv("generated_vs_original_with_beam.csv")

@st.cache_data
def load_rag():
    return pd.read_csv("generated_vs_original_RAG.csv")

@st.cache_data
def load_rag_infer():
    return pd.read_csv("generated_vs_original_RAG_infer.csv")

@st.cache_data
def load_genre_rag():
    return pd.read_csv("generated_vs_original_genre_RAG.csv")

@st.cache_data
def load_genre_only():
    return pd.read_csv("generated_vs_original_genre.csv")

df_meta = load_metadata()
df_base = load_baseline()
df_rag = load_rag()
df_rag_infer = load_rag_infer()
df_genre_rag = load_genre_rag()
df_genre_only = load_genre_only()

# Titles common to all three sources
valid_titles = set(df_meta["title"].str.lower()) \
    & set(df_base["Title"].str.lower()) \
    & set(df_rag["Title"].str.lower()) \
    & set(df_rag_infer["Title"].str.lower()) \
    & set(df_genre_rag["Title"].str.lower()) \
    & set(df_genre_only["Title"].str.lower())

valid_titles_sorted = sorted({title for title in df_meta["title"] if title.lower() in valid_titles})

# === Check if the poster URL is valid ===
def is_valid_url(url):
    try:
        r = requests.head(url)
        return r.status_code == 200
    except:
        return False

# === Fallback: Fetch poster from OMDb ===
def fetch_omdb_poster(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        res = requests.get(url).json()
        poster = res.get("Poster", "")
        if poster and poster != "N/A" and is_valid_url(poster):
            return poster
    except:
        pass
    return None


# Add movie genre to the metadata DataFrame


# === UI ===
st.title("🍿 Tagline Generator")


title_input = st.selectbox("Choose a movie:", valid_titles_sorted)

if title_input:
    match = df_meta[df_meta["title"].str.lower() == title_input.strip().lower()]
    
    if not match.empty:
        row = match.iloc[0]
        title = row["title"]
        overview = row["overview"]
        original_tagline = row["tagline"]
        poster_path = row.get("poster_path", "")
        backdrop_path = row.get("backdrop_path", "")
        genre = row.get("parsed_genres", "N/A")

        # Construct URLs for posters
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if pd.notna(poster_path) else ""
        backdrop_url = f"https://image.tmdb.org/t/p/w500{backdrop_path}" if pd.notna(backdrop_path) else ""
        poster_displayed = False

        # Try poster url, backdrop url, OMDb
        if is_valid_url(poster_url):
            st.markdown(
                f"<div style='text-align: center;'>"
                f"<img src='{poster_url}' width='300'><br>"
                f"<span style='font-size: 14px;'>Poster</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            poster_displayed = True

        elif is_valid_url(backdrop_url):
            st.markdown(
                f"<div style='text-align: center;'>"
                f"<img src='{backdrop_url}' width='300'><br>"
                f"<span style='font-size: 14px;'>Poster</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            poster_displayed = True

        else:
            omdb_poster = fetch_omdb_poster(title)
            if omdb_poster:
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<img src='{omdb_poster}' width='300'><br>"
                    f"<span style='font-size: 14px;'>Poster from OMDb</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                poster_displayed = True


        if not poster_displayed:
            st.warning("No poster available.")

        # Look up generated taglines
        base_row = df_base[df_base["Title"].str.lower() == title.lower()]
        rag_row = df_rag[df_rag["Title"].str.lower() == title.lower()]
        rag_infer_row = df_rag_infer[df_rag_infer["Title"].str.lower() == title.lower()]

        base_gen = base_row.iloc[0]["Generated"] if not base_row.empty else "❌ Not found"
        rag_gen = rag_row.iloc[0]["Generated"] if not rag_row.empty else "❌ Not found"
        rag_infer_gen = rag_infer_row.iloc[0]["Generated"] if not rag_infer_row.empty else "❌ Not found"

        genre_rag_row = df_genre_rag[df_genre_rag["Title"].str.lower() == title.lower()]
        genre_only_row = df_genre_only[df_genre_only["Title"].str.lower() == title.lower()]

        genre_rag_gen = genre_rag_row.iloc[0]["Generated"] if not genre_rag_row.empty else "❌ Not found"
        genre_only_gen = genre_only_row.iloc[0]["Generated"] if not genre_only_row.empty else "❌ Not found"

        # ----------- Display text sections -----------
        # Genre
        st.subheader("Genre")
        genre_tags = "".join([
            f"<span style='background-color: #f45555; color: white; padding: 4px 10px; margin-right: 5px; border-radius: 12px; font-size: 14px;'>{g.strip()}</span>"
            for g in genre.split(",")
        ])

        st.markdown(genre_tags, unsafe_allow_html=True)

        # Overview
        st.subheader("Overview")
        st.write(overview)

        # Original tagline
        st.subheader("Original Tagline")
        st.write(original_tagline)

        # Baseline tagline
        st.subheader("Baseline Model Generated Tagline")
        st.markdown(
            f"""
            <div style='
                background-color: #fff8e1;
                padding: 15px;
                border-left: 5px solid #ffc107;
                border-radius: 5px;
                font-size: 16px;
                color: #333;
            '>
                {base_gen}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # RAG train and infer tagline
        st.subheader("RAG At Both Training and Inference Generated Tagline")
        st.markdown(
            f"""
            <div style='
                background-color: #e6f0ff;
                padding: 15px;
                border-left: 5px solid #3399ff;
                border-radius: 5px;
                font-size: 16px;
                color: #111;
            '>
                {rag_gen}
            </div>
            """,
            unsafe_allow_html=True
        )

        # RAG at infer only tagline
        st.subheader("RAG At Inference Only Generated Tagline")
        st.markdown(
            f"""
            <div style='
                background-color: #f0f8e1;
                padding: 15px;
                border-left: 5px solid #71bc78;
                border-radius: 5px;
                font-size: 16px;
                color: #111;
            '>
                {rag_infer_gen}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Overview + Genre Model Generated Tagline")
        st.markdown(
            f"""
            <div style='
                background-color: #e8f8ff;
                padding: 15px;
                border-left: 5px solid #00bcd4;
                border-radius: 5px;
                font-size: 16px;
                color: #111;
            '>
                {genre_only_gen}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Overview + Genre + RAG Model Generated Tagline")
        st.markdown(
            f"""
            <div style='
                background-color: #f3e8ff;
                padding: 15px;
                border-left: 5px solid #b266ff;
                border-radius: 5px;
                font-size: 16px;
                color: #111;
            '>
                {genre_rag_gen}
            </div>
            """,
            unsafe_allow_html=True
        )


