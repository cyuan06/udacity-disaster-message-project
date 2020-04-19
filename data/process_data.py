"""
Processing Data

Arguments:
    1. csv file containing messages
    2. csv file containing categories
    3. SQLITE database that stores cleaned datasets
"""



import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This Function loads datasets from file and merges two datasets into one

    Arguments:
        messages_filepath: path of messages.csv
        categories_filepath: path of categories.csv

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how = 'inner', on = 'id')
    return df

def clean_data(df):
    """
    Clean Data Function

    Arguments:
        df: data loaded from load_data function
    """
    #create dataframe called categories to store 36 seperate category columns
    categories = df['categories'].str.split(";", expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x:x.split('-')[0]))
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x.split('-')[1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype("int64")
        
    #drop the original columns
    df = df.drop(columns ='categories')
    df = df.drop(columns = 'original')
    #replace any 2 with 1 in the related column
    categories['related'].replace(2,1,inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    #drop na values
    df = df.dropna(axis = 0)
    return df
def save_data(df, database_filename):
    """
    Save Data Function

    Arguments:
        df: data that loaded by load_function and then cleand by clean_data function
        database_filename: database file path
    """
    #save data to database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterMessage', engine, index=False)


def main():
    """
    ETL function

    1. Data extraction from csv files
    2. Data cleaning and processing
    3. load data to sqlite database

    """
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