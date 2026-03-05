import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Create text field
movies["text"] = movies["overview"].fillna("") + " " + movies["genres"].fillna("")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(movies["text"].tolist())

# Similarity matrix
similarity_matrix = cosine_similarity(embeddings)


# Recommendation function
def recommend_with_explanation(movie_title, top_n=5):

    if movie_title not in movies["title"].values:
        return []

    idx = movies[movies["title"] == movie_title].index[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i in scores[1:top_n+1]:
        movie = movies.iloc[i[0]]

        explanation = f"Recommended because it shares similar genres ({movie['genres']}) with {movie_title}."

        results.append({
            "title": movie["title"],
            "explanation": explanation
        })

    return results


# UI
st.title("🎬 Movie Recommendation System")

movie = st.text_input("Enter a movie title (example: Avatar)")

if movie:

    results = recommend_with_explanation(movie)

    if results:

        st.subheader("Recommended Movies")

        for r in results:

            st.write("###", r["title"])
            st.write(r["explanation"])

    else:
        st.write("Movie not found.")