from flask import Flask, redirect, url_for, render_template, request, jsonify
import json
import pickle
import pandas as pd
import numpy as np
import requests
from flask_cors import CORS
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

model = pickle.load(open('finalmodel.pkl', 'rb'))
csr_data = pickle.load(open('csr_data.pkl', 'rb'))
movie_lst = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movie_lst)
ratings = pd.read_csv("ratings.csv")
rating_list = pickle.load(open('final_ratings.pkl', 'rb'))
final_ratings = pd.DataFrame(rating_list)

knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=20)
knn.fit(csr_data)


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/started', methods=['Post', 'Get'])
def started():
    res = "reccomend"
    return render_template(url_for(res))


@app.route('/reccomend.html')
def reccomend():
    return render_template('reccomend.html')


def Convert(a):
    it = iter(a)
    res_dct = dict(zip(it, it))
    return res_dct


@app.route('/finalresult', methods=['POST', 'GET'])
def get_movie_recommendation():
    data = request.get_json()
    movie_name = data['movie']
    n_movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_ratings[final_ratings['movieId']
                                  == movie_idx].index[0]
        distances, indices = knn.kneighbors(
            csr_data[movie_idx], n_neighbors=n_movies_to_recommend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(
        ), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        recommend = []
        for val in rec_movie_indices:
            movie_idx = final_ratings.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend.append({
                'Title': movies.iloc[idx]['title'].values[0],
                'Distance': val[1]
            })
        df = pd.DataFrame(recommend, index=range(1, n_movies_to_recommend+1))
        finallist = df['Title'].tolist()
        # array1 = np.array(finallist)
        # return finallist
        rmovies = json.dumps(finallist)
        return rmovies
    else:
        return "No movie found with that name. Please check your input."


@app.route('/test', methods=['GET'])
def test():
    return ("movie")


if __name__ == "_main_":
    app.run(debug=True)
