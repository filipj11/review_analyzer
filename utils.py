import pandas as pd
from joblib import dump, load
import nltk


def read_csv(filepath):
    return pd.read_csv(filepath)


def load_model(filepath):
    return load(filepath)


def save_model(model, filepath):
    dump(model, filepath)


def download_nltk_packages():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')






