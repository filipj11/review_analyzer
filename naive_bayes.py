from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from utils import read_csv, save_model
from preprocessing import full_preprocess


def train_naive_bayes(X, y):
    clf = GaussianNB()
    clf.fit(X, y)
    return clf


def training_pipeline():
    dataset_filepath = Path.cwd() / 'fake reviews dataset.csv'
    model_filepath = Path.cwd() / 'fake_review_NB.joblib'

    dataset_df = read_csv(dataset_filepath)
    preprocessed_corpus = full_preprocess(dataset_df['text_'])

    vectorizer = TfidfVectorizer(tokenizer=lambda s:s, lowercase=False, ngram_range=(1, 2), min_df=0.05)
    X = vectorizer.fit_transform(preprocessed_corpus)

    y = dataset_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.33)

    fake_review_clf = train_naive_bayes(X_train, y_train)
    y_pred = fake_review_clf.predict(X_test)

    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    save_model(fake_review_clf, model_filepath)


if __name__ == '__main__':
    training_pipeline()


