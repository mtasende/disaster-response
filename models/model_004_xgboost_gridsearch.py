from models.model import Model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class Model004(Model):
    """
    This model uses a simple NLP pipeline and an XGBoost classifier.
    It performs Grid Search on some parameters.
    """

    def build_model(self):
        """ Build a pipeline to preprocess and classify text. """
        pipeline = Pipeline([
            ('vec', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(XGBClassifier(random_state=2018)))
        ])
        self.model = pipeline
        return pipeline

    def tune_params(self, X_train, Y_train):
        """ Grid search for better parameters. Return the best model found. """
        parameters = {
            'tfidf__smooth_idf': [True, False],
            'clf__estimator__max_depth': [3, 6, 20],
            'clf__estimator__subsample': [0.5, 1.0]
        }
        cv = GridSearchCV(self.model, parameters, n_jobs=-1)
        cv.fit(X_train, Y_train)
        return cv.best_estimator_
