import streamlit as st
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")

movies["text"] = (
    movies["overview"].fillna("") + " " +
    movies["genres"].fillna("") + " " +
    movies["keywords"].fillna("")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(movies["text"].tolist())

similarity_matrix = cosine_similarity(embeddings)


def recommend_with_explanation(movie_title, top_n=5):

    if movie_title not in movies["title"].values:
        return []

    idx = movies[movies["title"] == movie_title].index[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i in scores[1:top_n+1]:

        movie = movies.iloc[i[0]]

        try:
            genres = ", ".join([g["name"] for g in ast.literal_eval(movie["genres"])])
        except:
            genres = movie["genres"]

        explanation = f"Recommended because it shares similar genres ({genres}) with {movie_title}."

        results.append({
            "title": movie["title"],
            "explanation": explanation
        })

    return results

def recommend_by_preference(user_text, top_n=5):

    user_embedding = model.encode([user_text])

    scores = cosine_similarity(user_embedding, embeddings)[0]

    top_indices = scores.argsort()[::-1][:top_n]

    results = []

    for i in top_indices:

        movie = movies.iloc[i]

        try:
            genres = ", ".join([g["name"] for g in ast.literal_eval(movie["genres"])])
        except:
            genres = movie["genres"]

        explanation = f"This movie matches your preference '{user_text}' and belongs to genres: {genres}."

        results.append({
            "title": movie["title"],
            "explanation": explanation
        })

    return results

st.title("🎬 Movie Recommendation System")

movie = st.text_input("Enter a movie title (example: Avatar)")
preference = st.text_input("Or describe what movies you like (example: space sci-fi movies)")

if movie:

    results = recommend_with_explanation(movie)

    if results:

        st.subheader("Recommended Movies")

        for r in results:

            st.write("###", r["title"])
            st.write(r["explanation"])

    else:
        st.write("Movie not found.")

elif preference:

    results = recommend_by_preference(preference)

    st.subheader("Recommended Movies")

    for r in results:
        st.write("###", r["title"])
        st.write(r["explanation"])