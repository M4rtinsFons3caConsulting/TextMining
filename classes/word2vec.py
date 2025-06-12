from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Word2Vec(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.tokenizer = str.split
        self.vector_size = self.model.vector_size


    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        return np.array([self._vectorize(doc) for doc in X])
    

    def _vectorize(self, doc):
        tokens = self.tokenizer(doc)
        valid_tokens = [token for token in tokens if token in self.model.key_to_index]
        if not valid_tokens:
            return np.zeros(self.vector_size)
        
        return np.mean([self.model[token] for token in valid_tokens], axis=0)