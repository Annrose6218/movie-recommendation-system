import streamlit as st
import requests
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore

# Function to fetch movie poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return full_path
    else:
        return "https://via.placeholder.com/500x750?text=Poster+Not+Available"

# Load the dataset
df = pd.read_excel('tmdb.xlsx')

# Create mappings from IDs to integer indices
movie_ids = df['id'].unique()
movie_to_index = {movie: idx for idx, movie in enumerate(movie_ids)}
df['movie_index'] = df['id'].map(movie_to_index)

# Prepare input data
X = df[['movie_index']].values
y = df['vote_average'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters
num_movies = len(movie_ids)
embedding_dim = 50

# Inputs
movie_input = Input(shape=(1,), name='movie_input')

# Embeddings
movie_embedding = Embedding(num_movies, embedding_dim)(movie_input)

# Flatten embeddings
movie_flat = Flatten()(movie_embedding)

# Dense layers
dense = Dense(128, activation='relu')(movie_flat)
output = Dense(1, activation='linear')(dense)  # Predicting rating

# Model
model = Model(inputs=movie_input, outputs=output)
model.compile(optimizer=Adam(), loss=MeanSquaredError())

# Summary
model.summary()

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
st.write(f"Test loss: {loss}")

# Map movie titles to indices
movie_id_map = {title: idx for idx, title in enumerate(df['title'].values)}

st.header("Movie Recommendation System")

# Create a dropdown to select a movie
selected_movie = st.selectbox("Select a movie:", df['title'].values)

def recommend(movie_title):
    movie_idx = movie_id_map.get(movie_title, -1)
    if movie_idx == -1:
        st.error("Movie not found.")
        return [], []

    # Predict ratings for all movies
    all_movie_indices = np.arange(num_movies).reshape(-1, 1)  # Reshape for model input
    predicted_ratings = model.predict(all_movie_indices).flatten()

    # Get top 5 recommendations
    top_indices = np.argsort(predicted_ratings)[-6:-1][::-1]
    recommended_movies = df['title'].values[top_indices].tolist()
    recommended_posters = [fetch_poster(df.iloc[idx].id) for idx in top_indices]

    return recommended_movies, recommended_posters

if st.button("Recommend"):
    movie_names, movie_posters = recommend(selected_movie)
    if not movie_names:  # Corrected check for an empty list
        st.error("No recommendations available.")
    else:
        cols = st.columns(len(movie_names))
        for col, name, poster in zip(cols, movie_names, movie_posters):
            with col:
                st.text(name)
                st.image(poster)
