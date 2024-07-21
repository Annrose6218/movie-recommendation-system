import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_excel('tmdb.xlsx')

# Preview the data
print(df.head())

# Assuming your dataset has columns 'user_id', 'movie_id', and 'rating'

movie_ids = df['id'].unique()

# Create mappings from IDs to integer indices

movie_to_index = {movie: idx for idx, movie in enumerate(movie_ids)}

# Convert IDs to indices

df['movie_index'] = df['id'].map(movie_to_index)

# Prepare input data
X = df[[ 'movie_index']].values
y = df['vote_average'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)