import pandas as pd
from sqlalchemy import create_engine
import re
from data.process_data import MESSAGES_TABLE

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib


class Model(object):
    """
    Implements the model interface.
    It also serves as the base class for more complex models.
    """

    def load_data(self, database_filepath):
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        df = pd.read_sql_table(MESSAGES_TABLE, engine)
        X = df.loc[:, 'message']
        y = df.iloc[:, 4:]
        category_names = y.columns.tolist()

        return X, y, category_names

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

        # Part-of-Speech Tagging
        tokens = [(token[0], self.get_wordnet_pos(token[1])) for token in pos_tag(tokens)]

        # Named Entity Recognition
        # TODO: Add this to the pipeline. The punctuation is important to recognize the
        # entities.

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(*token) for token in tokens]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return tokens

    def build_model(self):
        pipeline = Pipeline([
            ('vecfrom data.process_data import MESSAGES_TABLEt', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
        return pipeline

    def evaluate_model(self, model, X_test, Y_test, category_names):
        pass

    def save_model(self, model, model_filepath):
        joblib.dump(model, model_filepath)

    @staticmethod
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
