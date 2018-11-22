from models.model import Model
from sklearn.model_selection import GridSearchCV


class Model001(Model):
    """
    This is the first, and most basic, model.
    It uses a simple NLP pipeline and a RandomForest as a classifier.
    The only added feature with respect to the Model class is a Grid Search.
    """
    def tune_params(self, X_train, Y_train):
        """ Grid search for better parameters. Return the best model found. """
        parameters = {
            # 'features__text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__smooth_idf': [True, False],
            'clf__estimator__max_depth': [None, 7]
            # 'clf__estimator__n_estimators': [10, 100],
        }
        cv = GridSearchCV(self.model, parameters, n_jobs=-1)
        cv.fit(X_train, Y_train)
        return cv.best_estimator_
