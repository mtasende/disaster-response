from models.model import Model
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import re
import numpy as np

PRE_TRAINED_W2V_PATH = '../data/GoogleNews-vectors-negative300.bin'


class Model003(Model):
    """
    This is the first, and most basic, model.
    It uses a simple NLP pipeline and a RandomForest as a classifier.
    This class uses an XGBoost classifier instead of the default one.
    """

    def tokenize(self, text):
        """ Basic tokenization function. """
        # Case normalization
        temp_text = text.lower()

        # Punctuation removal
        temp_text = re.sub(r'[^a-zA-Z0-9]', ' ', temp_text)

        # Tokenize
        tokens = word_tokenize(temp_text)

        # Stop Word Removal
        stop_words = stopwords.words("english")
        tokens = [word for word in tokens if word not in stop_words]

        return tokens

    def build_model(self):
        pipeline = Pipeline([
            ('word2vec', Word2VecTransformer(
                filepath=PRE_TRAINED_W2V_PATH,
                tokenizer=self.tokenize
            )),
            ('clf', MultiOutputClassifier(XGBClassifier(random_state=2018)))
        ])
        self.model = pipeline
        return pipeline


class Word2VecTransformer(TransformerMixin, BaseEstimator):
    """Transforms words into word2vec vectors. """

    def __init__(self, filepath, tokenizer):
        super().__init__()
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.model = None

    def fit(self, X, y):
        """ Get Google's pre-trained word2vec model. """
        self.model = KeyedVectors.load_word2vec_format(self.filepath,
                                                       binary=True)
        return self

    def transform(self, X):
        """ Transform the sentences to vectors. """
        X_tokenized = [self.tokenizer(sentence) for sentence in X]
        X_tr = list()
        null_sentences_count = 0
        for sentence in X_tokenized:
            sentence_tr = list()
            for word in sentence:
                try:
                    sentence_tr.append(self.model[word])
                except KeyError:
                    pass
            if len(sentence_tr) == 0:
                null_sentences_count += 1
                X_tr.append(np.zeros(300))
            else:
                X_tr.append(np.mean(sentence_tr, axis=0))

        print('Found {} non-convertible sentences. '.format(
            null_sentences_count) + 'That is {}% of the total.'.format(
                  (100 * null_sentences_count/len(X_tokenized))
              ))

        return np.stack(X_tr, axis=0)
