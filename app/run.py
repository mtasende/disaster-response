import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine

import data_utils_mt.utils as utils
from models.model import Model

DATABASE_PATH = '../data/DisasterResponse.db'
TABLE_NAME = 'messages'
MODEL_FILEPATH = '../models/classifier.pkl'
N_SAMPLES = 1000


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///{}'.format(DATABASE_PATH))
df = pd.read_sql_table(TABLE_NAME, engine)

# load model
model = joblib.load(MODEL_FILEPATH)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # Graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Graph 2
    positive = df.iloc[:, 4:].sum()
    negative = df.shape[0] - positive
    positive = positive.sort_values(ascending=True)
    negative = negative.sort_values(ascending=True)

    # Graph 3
    clustered_labels, cluster_idx = utils.cluster_corr(df.iloc[:, 4:])
    correlations = clustered_labels.corr().dropna(
        how='all', axis=0).dropna(how='all', axis=1)

    # Graph 4
    container = Model()
    tokenized = [container.tokenize(text) for text in df.message.iloc[:N_SAMPLES]]
    tokenized_arr = np.concatenate(tokenized)
    values, counts = np.unique(tokenized_arr, return_counts=True)
    counts_df = pd.Series(counts, index=values).sort_values(ascending=False)
    counts_to_plot = counts_df.iloc[:20].sort_values(
        ascending=True) / counts_df.sum()

    # create visuals

    # Graph 1
    graph1 = {
        'data': [
            Bar(
                x=genre_names,
                y=genre_counts
            )
        ],

        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    }


    # Graph 2
    trace1 = Bar(
    y=positive.index.tolist(),
    x=positive.values.tolist(),
    name='Positive label',
    orientation='h'
    )

    trace2 = Bar(
    y=negative.index.tolist(),
    x=negative.values.tolist(),
    name='Negative label',
    orientation='h'
    )

    graph2 = {
        'data': [trace1, trace2],
        'layout': {
            'title': 'Distribution of Labels',
            'yaxis': {
                'title': "Label"
            },
            'xaxis': {
                'title': "Number of cases"
            },
            'barmode': 'stack'
        }
    }


    # Graph 3
    graph3 = {
        'data': [
            Heatmap(
                x=correlations.index.values,
                y=correlations.index.values,
                z=correlations.values
            )
        ],

        'layout': {
            'title': 'Clustered Correlations between Labels',
        }
    }


    # Graph 4
    graph4 = {
        'data': [
            Bar(
                y=counts_to_plot.index.tolist(),
                x=counts_to_plot.values.tolist(),
                name='Most frequent tokens',
                orientation='h'
                )
        ],
        'layout': {
            'title': 'Most frequent tokens',
            'yaxis': {
                'title': "Token"
            },
            'xaxis': {
                'title': 'Frequency (in a sample of {})'.format(N_SAMPLES)
            },
            'barmode': 'stack'
        }
    }


    graphs = [graph1, graph2, graph3, graph4]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    print(ids)
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
