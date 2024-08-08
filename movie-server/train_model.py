# train_model.py
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pickle

# File paths
movies_file_path = r'D:\movie-recommend\tmdb_5000_movies.csv\tmdb_5000_movies.csv'
credits_file_path = r'D:\movie-recommend\tmdb_5000_credits.csv\tmdb_5000_credits.csv'

# Load data
movies = pd.read_csv(movies_file_path)
credits = pd.read_csv(credits_file_path)

# Merge and prepare data
movie = movies.merge(credits, on='title')
movie = movie[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movie.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movie['genres'] = movie['genres'].apply(convert)
movie['keywords'] = movie['keywords'].apply(convert)
movie['cast'] = movie['cast'].apply(convert3)
movie['crew'] = movie['crew'].apply(fetch_director)
movie['overview'] = movie['overview'].apply(lambda x: x.split())
movie['keywords'] = movie['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movie['genres'] = movie['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movie['cast'] = movie['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movie['crew'] = movie['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movie['tags'] = movie['overview'] + movie['genres'] + movie['keywords'] + movie['cast'] + movie['crew']

new_df = movie[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# nltk porterstemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Save the trained components
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

new_df.to_csv('movies_with_tags.csv', index=False)
