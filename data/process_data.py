import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

MESSAGES_TABLE = 'messages'


def load_data(messages_filepath, categories_filepath):
    """ Get the messages and categories from CSV files. """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id', how='left')


def categories_split(df):
    """ Separate the categories in their own columns. """
    ohe_categories = pd.DataFrame(df.categories.str.split(';').apply(
        lambda x: {e.split('-')[0]: int(e.split('-')[1]) for e in x}).tolist())
    return df.join(ohe_categories).drop('categories', axis=1)


def clean_data(df):
    """ Prepare the data for ML use. """
    df = df.drop_duplicates().reset_index(drop=True)
    return categories_split(df)


def save_data(df, database_filename):
    """ Save the data to a sqlite database. """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(MESSAGES_TABLE, engine, index=False, if_exists='replace',
              chunksize=1000)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
