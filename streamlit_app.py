import streamlit as st
import pandas as pd
import requests
import os

# === OMDb API Key ===
OMDB_API_KEY = "a167b60b"  

# === Load metadata and generations ===
@st.cache_data
def load_metadata():
    df = pd.read_csv("movies_metadata.csv")
    df = df[df["overview"].notna() & df["tagline"].notna()]
    return df

@st.cache_data
def load_baseline():
    return pd.read_csv("generated_vs_original_with_beam.csv")

@st.cache_data
def load_rag():
    return pd.read_csv("generated_vs_original_RAG.csv")

df_meta = load_metadata()
df_base = load_baseline()
df_rag = load_rag()

# Titles common to all three sources
valid_titles = set(df_meta["title"].str.lower()) & set(df_base["Title"].str.lower()) & set(df_rag["Title"].str.lower())

valid_titles_sorted = sorted({title for title in df_meta["title"] if title.lower() in valid_titles})

# === Check if the image URL is valid ===
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

# === UI ===
st.title("🎬 Tagline Generator")

title_input = st.selectbox("Choose a movie title:", valid_titles_sorted)

if title_input:
    match = df_meta[df_meta["title"].str.lower() == title_input.strip().lower()]
    
    if not match.empty:
        row = match.iloc[0]
        title = row["title"]
        overview = row["overview"]
        original_tagline = row["tagline"]
        poster_path = row.get("poster_path", "")
        backdrop_path = row.get("backdrop_path", "")

        # Construct URLs
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

        base_gen = base_row.iloc[0]["Generated"] if not base_row.empty else "❌ Not found"
        rag_gen = rag_row.iloc[0]["Generated"] if not rag_row.empty else "❌ Not found"

        # Display text sections
        st.subheader("Overview")
        st.write(overview)

        st.subheader("Original Tagline")
        st.write(original_tagline)

        st.subheader("Baseline Model Generated Tagline")
        st.write(base_gen)

        st.subheader("RAG Model Generated Tagline")
        st.success(rag_gen)
        
