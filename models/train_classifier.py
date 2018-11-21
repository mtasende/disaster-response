import sys
from sklearn.model_selection import train_test_split
# from models.model import Model
from models.model_002_xgboost import Model002
from time import time

current_model = Model002()  # Change this to use another model


def load_data(database_filepath):
    return current_model.load_data(database_filepath)


def build_model():
    return current_model.build_model()


def tune_params(model, X_train, Y_train):
    return current_model.tune_params(X_train, Y_train)


def evaluate_model(model, X_test, Y_test, category_names):
    current_model.evaluate_model(model, X_test, Y_test, category_names)


def save_model(model, model_filepath):
    current_model.save_model(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Hyperparameter Tuning...')
        tic = time()
        model = tune_params(model, X_train, Y_train)
        toc = time()
        print('Hyperparameter tuning time: {} seconds'.format(toc - tic))

        print('Training model...')
        tic = time()
        model.fit(X_train, Y_train)
        toc = time()
        print('Training time: {} seconds'.format(toc - tic))

        print('Evaluating model...')
        tic = time()
        evaluate_model(model, X_test, Y_test, category_names)
        toc = time()
        print('Evaluation time: {} seconds'.format(toc - tic))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
