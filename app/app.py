import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")

# ------------------------------
# Load data
# ------------------------------
try:
    train_df = pd.read_csv(TRAIN_FILE)
except FileNotFoundError:
    st.error(f"File not found: {TRAIN_FILE}")
    st.stop()

# ------------------------------
# Prepare user-item matrix
# ------------------------------
user_item_matrix = train_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# ------------------------------
# Train SVD model
# ------------------------------
svd = TruncatedSVD(n_components=5, random_state=42)
matrix_factors = svd.fit_transform(user_item_matrix)
pred_matrix = np.dot(matrix_factors, svd.components_)
pred_df = pd.DataFrame(pred_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

# ------------------------------
# Streamlit Interface
# ------------------------------
st.title("Recommendation System Demo")
st.markdown("Enter a User ID to get top 5 recommended items.")

# User input
min_user = int(train_df.userId.min())
max_user = int(train_df.userId.max())
user_id = st.number_input("User ID", min_value=min_user, max_value=max_user, step=1)

if st.button("Get Recommendations"):
    if user_id in pred_df.index:
        user_ratings = pred_df.loc[user_id]
        # Exclude already rated items
        already_rated = train_df[train_df.userId == user_id]['movieId'].tolist()
        recommendations = user_ratings.drop(already_rated, errors='ignore').dropna().sort_values(ascending=False).head(5)
        
        if len(recommendations) > 0:
            st.subheader(f"Top 5 Recommendations for User {user_id}")
            for i, movie_id in enumerate(recommendations.index, 1):
                st.write(f"{i}. Movie ID: {movie_id} (Predicted Rating: {recommendations[movie_id]:.2f})")
        else:
            st.write("No recommendations available for this user.")
    else:
        st.write("User ID not found in the dataset.")
