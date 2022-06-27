"""
A project to obtain steam reviews and analyse sentiment

"""

import requests
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# for testing/debugging
pd.set_option('display.max_columns', 200)


def get_reviews(appid: int, num_reviews: int):
    base_url = "https://store.steampowered.com/appreviews/" + str(appid)
    params = {'json': 1,
              'filter': 'recent',
              'language': 'english',
              'cursor': '*',
              'purchase_type': 'all',
              'num_per_page': min(100, num_reviews)
              }
    reviews = []

    while num_reviews > 0:
        response = requests.get(base_url, params=params).json()
        reviews += response['reviews']
        num_reviews -= response['query_summary']['num_reviews']

        params['cursor'] = response['cursor']  # change the cursor parameter to load the next 100 reviews
        params['num_per_page'] = min(100, num_reviews)

        if response['query_summary']['num_reviews'] < 100: break

    return reviews


def reviews_to_df(reviews: list):
    df = pd.json_normalize(reviews)
    df.columns = df.columns.str.replace('author.', '', regex=False)
    return df


def raw_frequency_count(series, stop_words):
    counter = nltk.FreqDist()
    for r in series:
        word_list = [w.casefold() for w in word_tokenize(r) if w.casefold() not in stop_words]  # remove stopwords
        word_list = [w for w in word_list if w.isalpha()]  # keep only words made of letters
        counter.update(word_list)
    return counter


def lem_frequency_count(series, stop_words, do_not_lem=[]):
    counter = nltk.FreqDist()
    wnl = WordNetLemmatizer()
    for r in series:
        word_list = [w.casefold() for w in word_tokenize(r) if w.casefold() not in stop_words]  # remove stopwords
        word_list = [w for w in word_list if w.isalpha()]  # keep only words made of letters
        word_list = [wnl.lemmatize(w) if w not in do_not_lem else w for w in word_list]
        counter.update(word_list)
    return counter


def bigram_count(series, stop_words, do_not_lem=[]):
    counter = nltk.FreqDist()
    wnl = WordNetLemmatizer()
    for r in series:
        word_list = [w.casefold() for w in word_tokenize(r) if w.casefold() not in stop_words]  # remove stopwords
        word_list = [w for w in word_list if w.isalpha()]  # keep only words made of letters
        word_list = [wnl.lemmatize(w) if w not in do_not_lem else w for w in word_list]  # lemmatize
        bgs = nltk.bigrams(word_list)  # convert word_list to bigrams
        counter.update(bgs)
    return counter


def extract_features(df):
    feature_list = []
    classification_dict = {True: "pos",
                           False: "neg"}

    for row in df.itertuples():
        features = {"vader_compound": row.vader_compound,
                    "vader_positive": row.vader_pos,
                    "playtime": row.playtime_at_review}

        classification = classification_dict[row.voted_up]

        feature_list.append((features, classification))

    return feature_list
