import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Load & Preprocess Data ------------------
# Load CSV with proper column names and data types
df = pd.read_csv(
    "data.csv",
    names=['user_id', 'product_id', 'rating', 'timestamp'],
    header=0,
    dtype={"user_id": str, "product_id": str}
)

# Filter: Top 10 users and top 20 products (for demo)
top_users = df['user_id'].value_counts().head(10).index
top_products = df['product_id'].value_counts().head(20).index
df = df[df['user_id'].isin(top_users) & df['product_id'].isin(top_products)]

# Pivot ratings into matrix
ratings_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# ------------------ Build Recommendation Function ------------------
# Calculate cosine similarity between users
user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

def recommend_products(selected_user, top_n=5):
    similar_users = user_similarity_df[selected_user].sort_values(ascending=False)[1:]
    weighted_scores = pd.Series(dtype=float)

    for other_user, similarity_score in similar_users.items():
        weighted_scores = weighted_scores.add(
            ratings_matrix.loc[other_user] * similarity_score,
            fill_value=0
        )
    
    already_rated = ratings_matrix.loc[selected_user][ratings_matrix.loc[selected_user] > 0].index
    recommendations = weighted_scores.drop(already_rated).sort_values(ascending=False).head(top_n)
    
    return recommendations.index.tolist()

# ------------------ Streamlit UI ------------------
st.title("🛍️ Product Recommendation Demo")
st.markdown("Select a user to get personalized product recommendations based on other similar users.")

selected_user = st.selectbox("Choose a User ID", ratings_matrix.index)

if st.button("Recommend Products"):
    recommended = recommend_products(selected_user)
    if recommended:
        st.success(f"Top Recommended Products for User {selected_user}:")
        for pid in recommended:
            st.write(f"🔹 Product ID: `{pid}`")
    else:
        st.warning("No recommendations available for this user.")
