import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
user_item_matrix = train_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

svd = TruncatedSVD(n_components=5, random_state=42)
matrix_factors = svd.fit_transform(user_item_matrix)
pred_matrix = np.dot(matrix_factors, svd.components_)
pred_df = pd.DataFrame(pred_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

st.title("Recommendation System Demo")

user_id = st.number_input("Enter User ID", min_value=int(train_df.userId.min()), max_value=int(train_df.userId.max()), step=1)

if st.button("Get Recommendations"):
    if user_id in pred_df.index:
        user_ratings = pred_df.loc[user_id]
        already_rated = train_df[train_df.userId==user_id]['movieId'].tolist()
        recommendations = user_ratings.drop(already_rated).dropna().sort_values(ascending=False).head(5)
        st.write("Top 5 Recommendations for User", user_id)
        st.write(recommendations.index.tolist())
    else:
        st.write("User ID not found in data")
