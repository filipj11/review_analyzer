import pandas as pd
import pickle
import nltk


def read_csv(filepath):
    return pd.read_csv(filepath)


def load_model(filepath):
    return pickle.load(filepath)


def save_model(model, filepath):
    pickle.dump(model, filepath)


def download_nltk_packages():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')






