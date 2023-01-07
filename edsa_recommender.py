



"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import requests, json
from recommenders.collaborative_based import collab_model

# Importing data
movies = pd.read_csv('resources/data/movies_metadata.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')

# choose subset of dataset
data = movies.head(1000)
#list of movies
title_list = data['title'].values


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    response = requests.get(url,verify=False)
    json_data = response.json()
    poster_path = json_data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movie_names = []
    #data = data_preprocessing(27000)
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer(stop_words='english')
    count_matrix = count_vec.fit_transform(data['combined_features'])
    vectors = pd.DataFrame(count_matrix.toarray(), index=data['combined_features'].index.tolist())
    # cosine similarities
    cosine_sim = cosine_similarity(vectors, vectors)
    indices = pd.Series(data['title'])
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    # Store movie names
    recommended_movie_names = []
    recommended_movie_posters = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        movie_id = data['tmdbId'][i]
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(list(data['title'])[i])
    
    return recommended_movie_names,recommended_movie_posters

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    st.set_page_config(page_title='Movie Recommender Engine',page_icon=':movie_camera:',)
    page_options = ["Recommender System","How it works"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "How it works":
        # Header contents
        st.write('# How does this recommendation system work?')
        st.markdown('View full code on [GitHub](https://github.com/cliffordsepato/movie-recomendation-app)')
        st.markdown('These recommenders systems were built using `Python` and a special version of the [MovieLens](https://movielens.org/) dataset enriched with additional data, and resampled for fair evaluation purposes in order to accurately predict how users will rate a movie they have not yet viewed based on their historical preferences.')
        st.markdown('The recommendations are returned as a movie poster fetched using an API from the [themoviedb.org](https://www.themoviedb.org/documentation/api). ')
        st.markdown('The recommendation system can be divided into two parts:')
        st.markdown('* **Content-Based Filtering**: suggests similar items based on a particular item. This system uses item metadata, such as genre, director, plot keywords and actors for movies, to make these recommendations. The general idea behind this recommender systems is that if a person likes a particular item, he or she will also like an item that is similar to it. And to recommend that, it will make use of the users past item metadata. A good example could be YouTube, where based on your history, it suggests you new videos that you could potentially watch.') 
        st.markdown('* **Collaborative Filtering**:  this recommender attempts to predict the rating or preference that a user would give an item-based on past ratings and preferences of other users. Collaborative filters do not require item metadata like its content-based counterparts. The model is based on Singular Value Decomposition (SVD) Matrix Factorization method to generate latent features of movies.') 
    
        

    if page_selection == "Recommender System":
        # Header contents
        st.image('resources/imgs/animated_header_2.gif',use_column_width=True)
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[0:1000])
        movie_2 = st.selectbox('Second Option',title_list[1:1000])
        movie_3 = st.selectbox('Third Option',title_list[2:1000])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Show recommendations"):
                with st.spinner('Making the magic happen...:sunglasses:'):
                    recommended_movie_names,recommended_movie_posters = content_model(movie_list=fav_movies,top_n=10)    
                    st.title("Recommended for you:")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col6, col7, col8, col9, col10 = st.columns(5)
                with col1:
                    st.image(recommended_movie_posters[0])
                    st.caption(recommended_movie_names[0])
                with col2:
                    st.image(recommended_movie_posters[1])
                    st.caption(recommended_movie_names[1])

                with col3:
                    st.image(recommended_movie_posters[2])
                    st.caption(recommended_movie_names[2])
                with col4:
                    st.image(recommended_movie_posters[3])
                    st.caption(recommended_movie_names[3])
                with col5:
                    st.image(recommended_movie_posters[4])
                    st.caption(recommended_movie_names[4])
                with col6:
                    st.image(recommended_movie_posters[5])
                    st.caption(recommended_movie_names[5])
                with col7:
                    st.image(recommended_movie_posters[6])
                    st.caption(recommended_movie_names[6])

                with col8:
                    st.image(recommended_movie_posters[7])
                    st.caption(recommended_movie_names[7])
                with col9:
                    st.image(recommended_movie_posters[8])
                    st.caption(recommended_movie_names[8])
                with col10:
                    st.image(recommended_movie_posters[9])
                    st.caption(recommended_movie_names[9])

        if sys == 'Collaborative Based Filtering':
            if st.button("Show recommendations"):
                
                with st.spinner('Making the magic happen...:sunglasses:'):
                    top_recommendations,recommended_movie_posters = collab_model(movie_list=fav_movies,top_n=10)
                    st.title("Recommended for you:")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col6, col7, col8, col9, col10 = st.columns(5)
                with col1:
                    st.image(recommended_movie_posters[0])
                    st.caption(top_recommendations[0])
                with col2:
                    st.image(recommended_movie_posters[1])
                    st.caption(top_recommendations[1])

                with col3:
                    st.image(recommended_movie_posters[2])
                    st.caption(top_recommendations[2])
                with col4:
                    st.image(recommended_movie_posters[3])
                    st.caption(top_recommendations[3])
                with col5:
                    st.image(recommended_movie_posters[4])
                    st.caption(top_recommendations[4])
                with col6:
                    st.image(recommended_movie_posters[5])
                    st.caption(top_recommendations[5])
                with col7:
                    st.image(recommended_movie_posters[6])
                    st.caption(top_recommendations[6])

                with col8:
                    st.image(recommended_movie_posters[7])
                    st.caption(top_recommendations[7])
                with col9:
                    st.image(recommended_movie_posters[8])
                    st.caption(top_recommendations[8])
                with col10:
                    st.image(recommended_movie_posters[9])
                    st.caption(top_recommendations[9])

if __name__ == '__main__':
    main()
