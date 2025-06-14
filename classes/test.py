from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts a list of tokenized documents to their mean word embedding vectors.

    Parameters
    ----------
    word2vec_model : gensim.models.KeyedVectors or similar
        A pre-trained word2vec model with word vectors accessible via .wv[word].

    Attributes
    ----------
    word2vec_model : gensim.models.KeyedVectors
        The loaded word2vec model.
    dim : int
        Dimensionality of the word vectors.
    
    Methods
    -------
    fit(X, y=None)
        Does nothing and returns self. Included for compatibility with sklearn pipelines.
    transform(X)
        Transforms a list of tokenized documents into their mean embedding vectors.
    """

    def __init__(self, word2vec_model):
        # Initialize with a pre-trained word2vec model
        self.word2vec_model = word2vec_model
        
        # Store the dimensionality of word vectors for zero vector fallback
        self.dim = word2vec_model.vector_size

    def fit(self, X, y=None):
        # No fitting necessary, just return self to be sklearn compatible
        return self

    def transform(self, X):
        """
        Transform tokenized documents into mean word embedding vectors.

        Parameters
        ----------
        X : list of list of str
            List of documents, each document represented as a list of tokens.

        Returns
        -------
        np.ndarray
            2D array where each row is the mean vector of the words in a document.
            If none of the words are in the model, returns a zero vector.
        """
        return np.array([
            np.mean(
                [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
                or [np.zeros(self.dim)],  # Fallback to zero vector if no known words
                axis=0
            )
            for words in X
        ])
