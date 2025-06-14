from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import numpy as np
from typing import List, Optional, Union, Callable
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class W2VVectorizer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that converts a list of documents
    into document embeddings by averaging the Word2Vec vectors of their tokens.

    Parameters:
    -----------
    w2v_model : Word2Vec
        A trained Gensim Word2Vec model.
    """

    def __init__(
        self
        ,w2v_model: Word2Vec
    ):
        self.model = w2v_model
        self.vector_size = w2v_model.vector_size

    def fit(
        self
        ,X: Union[List[str], pd.Series]
        ,y: Optional[Union[List, pd.Series]] = None
    ) -> "W2VVectorizer":
        """
        No-op fit method to comply with scikit-learn's transformer interface.

        Parameters:
        -----------
        X : list or pandas.Series
            List of text documents.

        y : list or pandas.Series, optional
            Not used, exists for compatibility.

        Returns:
        --------
        self : W2VVectorizer
            Returns self.
        """
        return self

    def transform(
        self
        ,X: Union[List[str], pd.Series]
    ) -> np.ndarray:
        """
        Transforms each document into a fixed-size vector by averaging the Word2Vec vectors
        of the words in the document.

        Parameters:
        -----------
        X : list or pandas.Series
            List of text documents.

        Returns:
        --------
        X_transformed : np.ndarray
            Array of shape (n_samples, vector_size) containing document embeddings.
        """
        return np.vstack([self.document_vector(doc) for doc in X])

    def document_vector(
        self
        ,doc: str
    ) -> np.ndarray:
        """
        Converts a single document into a vector by averaging the Word2Vec vectors
        of its tokens. Out-of-vocabulary words are skipped.

        Parameters:
        -----------
        doc : str
            A text document.

        Returns:
        --------
        vec : np.ndarray
            The averaged Word2Vec vector of shape (vector_size,).
            Returns a zero vector if no words in the doc are in the vocabulary.
        """
        words = doc.split()
        word_vecs = [self.model.wv[word] for word in words if word in self.model.wv]

        if word_vecs:
            return np.mean(word_vecs, axis=0)
        else:
            return np.zeros(self.vector_size)


class CLSVectorizer(BaseEstimator, TransformerMixin):
    """
    Transformer that generates CLS embeddings for a list of text inputs using a pretrained embedding model.

    This class is designed to be compatible with scikit-learn pipelines. It expects the embeddings model
    to be a callable that takes a single string and returns a vector representation (either a numpy array or a tensor).

    Parameters
    ----------
    embeddings_model : Callable[[str], Union[np.ndarray, object]]
        A pretrained embedding model that takes a single text string as input and returns its embedding.
        The returned embedding should be either a numpy array or a tensor-like object with a `.numpy()` method.
    
    desc : str, optional (default="Generating CLS Embeddings")
        Description string for tqdm progress bar display during transformation.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer on the data. Does nothing for pretrained embeddings.
    transform(X)
        Transforms a list of text inputs into their corresponding CLS embeddings.
    """

    def __init__(
        self, embeddings_model: callable
    ) -> None:
        self.embeddings_model = embeddings_model

    def fit(
        self
        ,X: Optional[List[str]]
        ,y: Optional[List] = None
    ) -> "CLSVectorizer":
        """
        Fit method is required by scikit-learn but not used here since embeddings_model is pretrained.

        Parameters
        ----------
        X : list of str, optional
            List of text inputs (not used).
        y : list, optional
            Target values (not used).

        Returns
        -------
        self : CLSVectorizer
            Returns self.
        """
        return self

    def transform(
        self
        ,X: List[str]
    ) -> np.ndarray:
        """
        Generate CLS embeddings for a list of input texts.

        Parameters
        ----------
        X : list of str
            List of text inputs to transform.

        Returns
        -------
        np.ndarray
            A 2D numpy array where each row is the CLS embedding vector of the corresponding input text.
        """
        cls_embeddings = []
        for text in tqdm(X):
            emb = self.embeddings_model(text)  # Get embedding vector
            if hasattr(emb, "numpy"):
                emb = emb.numpy()  # Convert tensor to numpy array if necessary
            cls_embeddings.append(emb)
        
        # Stack all embeddings into a 2D array
        return np.vstack(cls_embeddings)
    

class BERTVectorizer:
    """
    A unified vectorizer that generates CLS token embeddings from any Hugging Face Transformer model.

    This class supports models like DistilBERT, RoBERTa, FinBERT, etc., as long as they follow the
    standard Hugging Face format and provide CLS token outputs.

    Parameters
    ----------
    model_name : str
        The name or path of the pretrained model (e.g., 'distilbert-base-uncased', 
        'cardiffnlp/twitter-roberta-base-sentiment', 'yiyanghkust/finbert-tone').
    
    Methods
    -------
    embed(text: str) -> torch.Tensor
        Returns the CLS embedding tensor for a single input text.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize the tokenizer and model from the provided Hugging Face model name.
        Sets the model to evaluation mode to disable dropout.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text: str) -> torch.Tensor:
        """
        Generate the CLS embedding for a single text input.

        Parameters
        ----------
        text : str
            Input text string to embed.

        Returns
        -------
        torch.Tensor
            A 1D tensor representing the CLS embedding of the input text,
            with shape (hidden_size,).
        """
        # Tokenize with appropriate padding/truncation and convert to tensor
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        # Inference mode (no gradients)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS embedding (first token's hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding.squeeze(0)
    