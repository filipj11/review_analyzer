import nltk
import string


class TextPreprocessor:
    def __init__(self, corpus):
        self.corpus = corpus

    def tokenize(self):
        self.corpus = [nltk.word_tokenize(item) for item in self.corpus]

    def remove_stopwords(self, language="english"):
        stopwords = nltk.corpus.stopwords.words(language)
        self.corpus = [[token for token in item if token not in stopwords] for item in self.corpus]

    def lowercase(self):
        self.corpus = [[token.lower() for token in item] for item in self.corpus]

    def remove_punctuation(self, punctuation):
        self.corpus = [[token for token in item if token not in punctuation] for item in self.corpus]

    def lemmatize(self):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        self.corpus = [[lemmatizer.lemmatize(token) for token in item] for item in self.corpus]


class PreprocessorManager:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def full_preprocessing(self):
        self.preprocessor.tokenize()
        self.preprocessor.lowercase()
        self.preprocessor.remove_punctuation(string.punctuation)
        self.preprocessor.lemmatize()
        self.preprocessor.remove_stopwords("english")


def full_preprocess(reviews):
    preprocessor = TextPreprocessor(reviews)
    preprocessor_manager = PreprocessorManager(preprocessor)
    preprocessor_manager.full_preprocessing()

    return preprocessor.corpus
