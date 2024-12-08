import streamlit as st
import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from transformers import pipeline
import urllib.request

# Initialize emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Dataset Configurations
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_FILE = "ml-100k.zip"
EXTRACT_DIR = "ml-100k"

# Load MovieLens Dataset
@st.cache_data
def load_movielens_data():
    if not os.path.exists(ZIP_FILE):
        st.info("Downloading dataset... Please wait.")
        urllib.request.urlretrieve(DATASET_URL, ZIP_FILE)

    if not os.path.exists(EXTRACT_DIR):
        with ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

    correct_path = os.path.join(EXTRACT_DIR, "ml-100k") if os.path.exists(os.path.join(EXTRACT_DIR, "ml-100k")) else EXTRACT_DIR

    movies_file = os.path.join(correct_path, "u.item")
    ratings_file = os.path.join(correct_path, "u.data")

    if not (os.path.exists(movies_file) and os.path.exists(ratings_file)):
        raise FileNotFoundError("Required dataset files not found. Please check the extraction.")

    movies_columns = ['movie_id', 'title']
    ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']

    movies = pd.read_csv(movies_file, sep='|', encoding='latin-1', names=movies_columns, usecols=[0, 1])
    ratings = pd.read_csv(ratings_file, sep='\t', names=ratings_columns, encoding='latin-1')

    merged_data = pd.merge(ratings, movies, on='movie_id')
    user_item_matrix = merged_data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
    user_item_sparse = csr_matrix(user_item_matrix.values)

    return movies, merged_data, user_item_matrix, user_item_sparse

# Recommendation Models
def knn_recommend(user_id, top_n, user_item_sparse, user_item_matrix):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_item_sparse)
    distances, indices = knn_model.kneighbors(user_item_sparse[user_id - 1], n_neighbors=top_n + 1)
    recommended_movies = user_item_matrix.columns[indices.flatten()[1:]]
    return recommended_movies.tolist()

def svd_recommend(user_id, top_n, user_item_sparse, user_item_matrix):
    U, sigma, Vt = svds(user_item_sparse, k=50)
    predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
    user_pred = predicted_ratings_df.loc[user_id].sort_values(ascending=False).head(top_n)
    return user_pred.index.tolist()

def random_forest_recommend(user_id, top_n, user_item_matrix):
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_item_reduced = svd.fit_transform(user_item_matrix)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = user_item_reduced
    y = user_item_matrix.values

    valid_ratings_mask = y > 0
    row_indices, col_indices = np.where(valid_ratings_mask)
    X_filtered = X[row_indices]
    y_filtered = y[valid_ratings_mask]

    rf_model.fit(X_filtered, y_filtered)
    user_index = user_id - 1
    user_features = user_item_reduced[user_index].reshape(1, -1)
    predicted_ratings = rf_model.predict(np.repeat(user_features, len(user_item_matrix.columns), axis=0))
    recommended_movies = user_item_matrix.columns[np.argsort(-predicted_ratings)[:top_n]]
    return recommended_movies.tolist()

# Streamlit Application
def main():
    st.title("🎥 Movie Recommender & Emotion Analysis")

    # Initialize session state
    if "selected_movies" not in st.session_state:
        st.session_state.selected_movies = []
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []

    # Load Data
    movies, merged_data, user_item_matrix, user_item_sparse = load_movielens_data()
    user_id = 1  # Fixed User ID Assumption

    # Model Selection
    selected_model = st.radio("Choose Model:", ["All Models", "KNN", "SVD", "Random Forest"])

    # Run Model Button
    if st.button("Run Model"):
        if selected_model == "All Models":
            knn_recs = knn_recommend(user_id, 10, user_item_sparse, user_item_matrix)
            svd_recs = svd_recommend(user_id, 10, user_item_sparse, user_item_matrix)
            rf_recs = random_forest_recommend(user_id, 10, user_item_matrix)
            st.session_state.recommendations = list(set(knn_recs + svd_recs + rf_recs))
        elif selected_model == "KNN":
            st.session_state.recommendations = knn_recommend(user_id, 10, user_item_sparse, user_item_matrix)
        elif selected_model == "SVD":
            st.session_state.recommendations = svd_recommend(user_id, 10, user_item_sparse, user_item_matrix)
        elif selected_model == "Random Forest":
            st.session_state.recommendations = random_forest_recommend(user_id, 10, user_item_matrix)

    # Select 5 Movies
    st.session_state.selected_movies = st.multiselect(
        "Select 5 Movies from Recommendations:", st.session_state.recommendations, default=st.session_state.selected_movies
    )

    # Show Analyze Emotions Button After Selection
    if len(st.session_state.selected_movies) == 5:
        st.subheader("Your Selected Movies")
        st.write(st.session_state.selected_movies)

        if st.button("Analyze Emotions"):
            st.subheader("**Emotion Analysis Results**")
            for movie in st.session_state.selected_movies:
                detailed_emotions = emotion_classifier(movie)
                sorted_emotions = sorted(detailed_emotions[0], key=lambda x: x['score'], reverse=True)

                st.write(f"**{movie}**")
                st.write("**Detailed Emotion Analysis:**")
                for emotion in sorted_emotions[:3]:
                    st.write(f"{emotion['label']}: {round(emotion['score'], 2)}")

if __name__ == "__main__":
    main()

