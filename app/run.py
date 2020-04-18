import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

class NounWordRatio(BaseEstimator, TransformerMixin):
    """Return the ratio of noun words"""

    def noun_ratio(self, text):
        # tokenize by word
        word = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(word)
        noun_length = 0
        #calculate the number of noun word
        for value in pos_tags:
            if value[1] in ['NN', 'NNP','NNS','NNPS']:
                noun_length += 1
        #validate the length of word    
        if len(word) > 0:
            return noun_length / len(word)
        else:
            return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply function to all values in X
        X_Cal = pd.Series(X).apply(self.noun_ratio)

        df_X_Cal = pd.DataFrame(X_Cal)
        #for any null values, fill zero
        return df_X_Cal.fillna(0)



def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessage', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #create another plot
    category_names = df.iloc[:,4:].columns
    category_value = (df.iloc[:,4:] != 0).sum().sort_values(ascending = False).values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
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
        },
            # GRAPH 2 - category graph    
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_value
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 30
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
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