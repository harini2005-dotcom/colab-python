# colab-python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie rating data (you can replace this with a real dataset)
# User ratings for movies, with ratings on a scale of 1-5
data = {
    'MovieID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'MovieName': ['The Matrix', 'Inception', 'The Dark Knight', 'Interstellar', 'Titanic', 'Avatar', 'The Avengers', 'The Lion King', 'Jurassic Park', 'The Godfather'],
}

# Create a DataFrame for movies
movies = pd.DataFrame(data)

# Sample user ratings (user IDs as index, movie IDs as columns)
ratings_data = {
    1: [5, 4, 0, 5, 3, 0, 4, 5, 0, 2],
    2: [4, 0, 4, 5, 0, 3, 5, 0, 2, 3],
    3: [3, 3, 4, 4, 5, 0, 3, 4, 2, 0],
    4: [5, 0, 0, 4, 4, 5, 0, 4, 3, 0],
}

# Create a DataFrame for ratings
ratings = pd.DataFrame(ratings_data, index=movies['MovieID'])

# Replace 0 ratings with NaN (we assume 0 means no rating)
ratings = ratings.replace(0, np.nan)

# Calculate cosine similarity between users based on movie ratings
cosine_sim = cosine_similarity(ratings.T.fillna(0))

# Create a DataFrame for user-user similarities
cosine_sim_df = pd.DataFrame(cosine_sim, index=ratings.columns, columns=ratings.columns)

# Function to recommend movies based on a given user (user_id)
def recommend_movies(user_id, top_n=3):
    # Get the list of movies that the user has rated
    user_ratings = ratings[user_id].dropna()

    # Get the similarity scores for this user with all other users
    user_similarities = cosine_sim_df[user_id]

    # Sort users by similarity score (most similar to least)
    similar_users = user_similarities.sort_values(ascending=False)

    # Create a list of recommended movies
    recommended_movies = []

    for similar_user_id in similar_users.index:
        if similar_user_id != user_id:
            similar_user_ratings = ratings[similar_user_id]
            # Filter out movies already rated by the user
            new_ratings = similar_user_ratings.dropna().loc[~similar_user_ratings.index.isin(user_ratings.index)]
            for movie, rating in new_ratings.items():
                recommended_movies.append((movie, rating))

        # Limit the number of recommendations
        if len(recommended_movies) >= top_n:
            break

    # Sort by rating to recommend the highest-rated movies
    recommended_movies = sorted(recommended_movies, key=lambda x: x[1], reverse=True)[:top_n]

    # Get movie names
    recommended_movie_names = [movies.loc[movies['MovieID'] == movie_id, 'MovieName'].values[0] for movie_id, _ in recommended_movies]

    return recommended_movie_names

# Example usage
user_id = 1
recommended = recommend_movies(user_id, top_n=3)
print(f"Recommended Movies for User {user_id}: {recommended}")


