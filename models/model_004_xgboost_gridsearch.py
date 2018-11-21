from models.model import Model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class Model004(Model):
    """
    This is the first, and most basic, model.
    It uses a simple NLP pipeline and a RandomForest as a classifier.
    This class uses an XGBoost classifier instead of the default one.
    """

    def build_model(self):
        pipeline = Pipeline([
            ('vec', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(XGBClassifier(random_state=2018)))
        ])
        self.model = pipeline
        return pipeline

    def tune_params(self, X_train, Y_train):
        parameters = {

        }
        cv = GridSearchCV(self.model, parameters, n_jobs=-1)
        cv.fit(X_train, Y_train)
        return cv.best_estimator_
