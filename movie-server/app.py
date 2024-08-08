# app.py
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained model components
with open('vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

new_df = pd.read_csv('movies_with_tags.csv')

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
    return recommended_movies

@app.route('/recommend', methods=['GET'])
def recommend_movies():
    movie_name = request.args.get('movie')
    if not movie_name:
        return jsonify({'error': 'No movie name provided'}), 400

    try:
        recommendations = recommend(movie_name)
        return jsonify({'recommendations': recommendations})
    except IndexError:
        return jsonify({'error': 'Movie not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
