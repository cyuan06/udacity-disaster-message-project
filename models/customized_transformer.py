
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])

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