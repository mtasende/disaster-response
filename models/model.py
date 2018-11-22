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
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score


class Model(object):
    """
    Implements the model interface.
    It also serves as the base class for more complex models.
    """

    def __init__(self):
        self.model = None

    def load_data(self, database_filepath):
        """
        Get the data from the database.

        Args:
            database_filepath(str): The path of the sqlite database.

        Returns:
            X(pandas.DataFrame): The input messages to classify.
            y(pandas.DataFrame): The desired output labels.
            category_names(list(str)): The names of the labels' categories.
        """
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
        tokens = [(token[0], self.get_wordnet_pos(token[1]))
                  for token in pos_tag(tokens)]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(*token) for token in tokens]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return tokens

    def build_model(self):
        """ Build a pipeline to preprocess and classify text. """
        pipeline = Pipeline([
            ('vec', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
        self.model = pipeline
        return pipeline

    def tune_params(self, X_train, Y_train):
        """ Grid search for better parameters. Return the best model found. """
        return self.model  # No hyper-parameter tuning

    def evaluate_model(self, model, X_test, Y_test, category_names):
        """
        Print some evaluation metrics:
            - Accuracy
            - Precision
            - Recall
            - F1-score
        """
        y_pred = model.predict(X_test)

        results = list()
        for i in range(y_pred.shape[1]):
            acc = accuracy_score(Y_test.values[:, i], y_pred[:, i])
            prec = precision_score(Y_test.values[:, i], y_pred[:, i],
                                   average='macro')
            rec = recall_score(Y_test.values[:, i], y_pred[:, i],
                               average='macro')
            f1 = f1_score(Y_test.values[:, i], y_pred[:, i], average='macro')
            results.append({'accuracy': acc,
                            'precision': prec,
                            'recall': rec,
                            'f1': f1})
        results_df = pd.DataFrame(results, index=category_names)
        print('-' * 100)
        print(results_df)
        print('-' * 100)
        print(results_df.describe())
        print('-' * 100)
        print('Main metric [f1-score]: {}'.format(results_df['f1'].mean()))
        print('-' * 100)

    def save_model(self, model, model_filepath):
        """ Save the model to a pickle. """
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
