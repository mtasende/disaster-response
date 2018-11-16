import pandas as pd
from sqlalchemy import create_engine
from data.process_data import MESSAGES_TABLE

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib


class Model(object):
    """
    Implements the most basic model.
    It also serves as the base class for more complex models.
    """

    def load_data(self, database_filepath):
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        df = pd.read_sql_table(MESSAGES_TABLE, engine)
        X = df.loc[:, 'message']
        y = df.iloc[:, 4:]
        category_names = y.columns.tolist()

        return X, y, category_names

    def tokenize(text):
        """ Basic tokenization function. """
        # Case normalization
        temp_text = text.lower()

        # Tokenize
        tokens = word_tokenize(temp_text)

        return tokens

    def build_model(self):
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
        return pipeline

    def evaluate_model(self, model, X_test, Y_test, category_names):
        pass

    def save_model(self, model, model_filepath):
        joblib.dump(model, model_filepath)

    def get_wordnet_pos(treebank_tag):
        """
        Transforms from Treebank tags to wordnet tags.
        As discussed here:
        https://stackoverflow.com/questions/15586721/
        wordnet-lemmatization-and-pos-tagging-in-python
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # If unknown, return the default value
