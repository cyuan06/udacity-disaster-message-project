"""
build up machine learning model

- To run ML pipeline that trains classifier and saves
 python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

 Arguments:
    1. SQLITE database path
    2. pickle filename to save ML model
"""
import sys
import pandas as pd
import numpy as np
import sqlalchemy
import sqlite3
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
import re
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load Data Function

    Arguments:
        database_filepath: path of SQLITE database db

    Output:
        X: Feature
        Y: Label
        category_names: column names of label
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("DisasterMessage", con = engine)
    X = df['message']
    #replace sign with empty space
    marker = re.compile(r'[^a-zA-Z]')
    X = X.replace(marker, " ")
    Y = df[['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize function

    Arguments:
        text: list of text messages
    Output:
        cleaned_tokens: cleaned tokenized text
    """
    #tokenize into words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    
    cleaned_tokens = []
    for tok in tokens:
        #lemmatize tokens
        cleaned_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        cleaned_tokens.append(cleaned_tok)
        
    #remove stop words
    cleaned_tokens = [word for word in cleaned_tokens if word not in stopwords.words('english')]
    
    return cleaned_tokens


class NounWordRatio(BaseEstimator, TransformerMixin):
    """
    Return the ratio of noun words
    
    Arguments:
        BaseEstimator: sklearn BaseEstimator class
        TransformerMixin: sklearn TransformerMixin class
    """
    def noun_ratio(self, text):
        """
        Return the noun words ratio of text

        Arguments:
            text: list of text messages

        Output:
            for each sample message return the noun words ratio
        """
        # tokenize by word
        word = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(word)
        noun_length = 0
        #calculate the number of noun word
        for value in pos_tags:
            #test if it's a noun
            if value[1] in ['NN', 'NNP','NNS','NNPS']:
                noun_length += 1
        #validate the length of word    
        if len(word) > 0:
            return noun_length / len(word)
        else:
            return 0

    def fit(self, x, y=None):
        """
        fit function
        """
        return self

    def transform(self, X):
        """
        Function to transform data
        """
        # apply function to all values in X
        X_Cal = pd.Series(X).apply(self.noun_ratio)

        df_X_Cal = pd.DataFrame(X_Cal)
        #for any null values, fill zero
        return df_X_Cal.fillna(0)


def build_model():
    """
    Build ML model

    This function intialize a machine learning pipeling for future training
    """
    pipeline = Pipeline([
    ('features', FeatureUnion([
            
        ('text_pipeline', Pipeline([
                 ('vect', CountVectorizer(tokenizer=tokenize)),
                 ('tfidf', TfidfTransformer())
            ])),
            
    ('NounWordRatio', NounWordRatio())
        ])),    
    ('RandomForest', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of ML model

    Arguments:
        model: ML pipeline model
        X_test: test datasets
        Y_test: test labels
        category_names: a column name list of test labels

    Output:
        print out classification reports for every feature to evaluate the performance of model
    """
    #predict test set
    y_pred = model.predict(X_test)
    # make into a dataframe for easier analysis
    df_pred = pd.DataFrame(y_pred, columns = category_names)
    # loop through all columns and print out the classification report
    for column in category_names:
        print('------------------------------------------------------\n')
        print('Feature_Name: {}\n'.format(column))
        print(classification_report(Y_test[column],df_pred[column]))


def save_model(model, model_filepath):
    """
    Save model into pickle file

    Arguments:
        model: ML pipeline model
        model_filepath: namepath to save model
    Output:
        saved model
    """
    #save model into file
    filename = model_filepath
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Train Classifier main function

    1. Extract data from SQLITE database
    2. Train ML model on training dataset
    3. Validate model performance on test dataset
    4. Save trained model into pickle file
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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