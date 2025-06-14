import re
from typing import Tuple, List, Any, Set, Optional, Union, Dict

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM
    )

import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Masking



def find_punctuated_tokens(texts: List[str]) -> Set[str]:
    """
    Identify and collect unique tokens containing punctuation from a list of texts.

    The function splits each text into tokens by whitespace and checks each token for any
    punctuation characters (anything except word characters and whitespace). Tokens containing
    the special character '‑' (Unicode non-breaking hyphen) are ignored and not added.

    Parameters
    ----------
    texts : List[str]
        A list of input strings to analyze.

    Returns
    -------
    Set[str]
        A set of unique tokens that contain punctuation (excluding tokens with '‑').
    """
    punctuated = set()  # To store unique tokens with punctuation

    for text in texts:
        tokens = text.split()  # Split text by whitespace into tokens

        for token in tokens:
            # Check if token contains any punctuation character
            if re.search(r'[^\w\s]', token):
                # Ignore tokens containing the non-breaking hyphen character '‑'
                if '‑' in token:
                    pass
                else:
                    punctuated.add(token)

    return punctuated


def eval_sklearn_model(
    vectorizer: BaseEstimator,
    classifier: ClassifierMixin,
    skf,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[List[Any], List[Any]]:
    """
    Evaluate a scikit-learn model pipeline with cross-validation.

    This function creates a pipeline consisting of a vectorizer and a classifier,
    then performs cross-validation splits provided by `skf` on the training data.
    For each fold, it trains the pipeline and predicts on the validation set,
    accumulating true and predicted labels across all folds.

    Parameters
    ----------
    vectorizer : BaseEstimator
        A feature extraction or transformation estimator (e.g., CountVectorizer, TfidfVectorizer).
    
    classifier : ClassifierMixin
        A scikit-learn compatible classifier (e.g., LogisticRegression, RandomForestClassifier).
    
    skf
        A cross-validator instance providing train/validation splits (e.g., StratifiedKFold).
    
    X_train : pd.DataFrame
        Training features.
    
    y_train : pd.Series
        Training labels.

    Returns
    -------
    Tuple[List[Any], List[Any]]
        Two lists containing the true labels and predicted labels aggregated from all validation folds.
    """
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    y_true_all = []
    y_pred_all = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        # Split data into current fold's training and validation sets
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train pipeline on training fold
        pipeline.fit(X_tr, y_tr)

        # Predict on validation fold
        y_pred = pipeline.predict(X_val)

        # Accumulate true and predicted labels
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

    return y_true_all, y_pred_all


def create_lstm_model(input_length: int, emb_size: int) -> Any:
    """
    Create and compile a Bi-directional LSTM model for multi-class classification.

    The model expects inputs shaped as (number of words per document, word embedding size).
    It uses a masking layer to ignore padding tokens (assumed to be zero vectors), followed
    by a bidirectional LSTM and a dense softmax output layer for 3-class classification.

    Parameters
    ----------
    input_length : int
        The fixed number of words (time steps) per input sequence.
    emb_size : int
        The dimensionality of the word embeddings.

    Returns
    -------
    keras.Model
        A compiled Keras model ready for training.
    """
    # Input layer: sequence of word embeddings
    input_ = Input(
        shape=(input_length, emb_size)
    )

    # Masking layer to ignore zero-padded embeddings
    mask = Masking(mask_value=0)(input_)

    # Bidirectional LSTM layer with 32 units
    lstm = Bidirectional(LSTM(units=32))(mask)

    # Output layer with 3 units and softmax activation for multi-class prediction
    output = Dense(3, activation='softmax')(lstm)

    # Define the model with inputs and outputs
    model = Model(inputs=input_, outputs=output)

    # Compile model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer='adam'
        ,loss='categorical_crossentropy'
        ,metrics=['accuracy']
    )

    return model


def embed_word2vec(
    texts: List[str]
    ,w2v_model
) -> List[np.ndarray]:
    """
    Convert a list of text strings into sequences of Word2Vec embeddings.

    Each text is tokenized by whitespace, and each token is mapped to its corresponding
    embedding vector from the provided Word2Vec model. Tokens not found in the model's
    vocabulary are skipped.

    Parameters
    ----------
    texts : List[str]
        List of input text strings to embed.
    w2v_model : gensim.models.Word2Vec
        Pretrained Word2Vec model with loaded embeddings.

    Returns
    -------
    List[np.ndarray]
        A list where each element is a numpy array of shape (sequence_length, embedding_dim),
        representing the embeddings for tokens in the corresponding input text.
        Sequences may vary in length depending on the number of known tokens per text.
    """
    sequences = []

    for text in texts:
        tokens = text.split()  # Simple whitespace tokenization

        # Get embeddings for tokens present in Word2Vec vocabulary
        vecs = [w2v_model.model.wv[word] for word in tokens if word in w2v_model.model.wv]

        # Convert list of vectors to numpy array for easier processing downstream
        sequences.append(np.array(vecs))

    return sequences


def embed_transformer(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    device: str = "cuda"
) -> List[np.ndarray]:
    """
    Generate embeddings for a list of texts using a transformer model.

    For each input text, this function tokenizes the text, feeds it through the
    transformer model, and extracts the last hidden states (token embeddings).
    The resulting embeddings are returned as numpy arrays.

    Parameters
    ----------
    texts : List[str]
        List of input text strings to embed.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer corresponding to the transformer model.
    model : transformers.PreTrainedModel
        Pretrained transformer model that returns last hidden states.
    device : str, optional (default="cuda")
        Device to run the model on ('cpu' or 'cuda').

    Returns
    -------
    List[np.ndarray]
        List of numpy arrays, each of shape (sequence_length, hidden_size),
        representing token embeddings for each input text.
    """
    sequences = []

    # Move model to specified device and set to evaluation mode
    model.to(device)
    model.eval()

    with torch.no_grad():
        for text in tqdm(texts):
            # Tokenize text with padding and truncation, return PyTorch tensors
            tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

            # Move input tensors to device (CPU/GPU)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Forward pass to get model outputs
            output = model(**tokens)

            # Extract token embeddings (last hidden state), remove batch dim, move to CPU, convert to numpy
            embeddings = output.last_hidden_state.squeeze(0).cpu().numpy()

            sequences.append(embeddings)

    return sequences


def eval_lstm_model(
    X_train: Union[List[str], 'pd.Series'],
    y_train: Union[List[int], 'pd.Series'],
    skf,
    emb_method: str,
    emb_model: Any,
    tokenizer: Optional[Any] = None,
    device: str = "cuda"
) -> Tuple[List[int], List[int], List[Any]]:
    """
    Train and evaluate an LSTM model using cross-validation with specified embeddings.

    Supports embedding methods: "word2vec" and "transformer". For each fold, the function
    generates embeddings for train and validation sets, pads sequences, one-hot encodes
    labels, trains the LSTM model, and collects true and predicted labels and training history.

    Parameters
    ----------
    X_train : List[str] or pd.Series
        Training texts.
    y_train : List[int] or pd.Series
        Integer-encoded training labels.
    skf : BaseCrossValidator
        Cross-validation splitter (e.g., StratifiedKFold).
    emb_method : str
        Embedding method to use: "word2vec" or "transformer".
    emb_model : Any
        Pretrained embedding model. For "word2vec" expects a Gensim Word2Vec model,
        for "transformer" expects a Hugging Face transformer model.
    tokenizer : Optional[Any], default=None
        Tokenizer required if emb_method is "transformer".
    device : str, default="cuda"
        Device to run transformer model on.

    Returns
    -------
    Tuple[List[int], List[int], List[Any]]
        y_true_all: List of true label indices across all folds.
        y_pred_all: List of predicted label indices across all folds.
        history_all: List of Keras History objects for each fold's training.
    """
    tf.random.set_seed(220)  # For reproducibility
    if device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    history_all: List[Any] = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr_texts, X_val_texts = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Generate embeddings depending on the selected method
        if emb_method == "word2vec":
            X_tr_vec = embed_word2vec(X_tr_texts, emb_model)
            X_val_vec = embed_word2vec(X_val_texts, emb_model)
            emb_size = emb_model.vector_size

        elif emb_method == "transformer":
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for transformer embeddings.")
            X_tr_vec = embed_transformer(X_tr_texts, tokenizer, emb_model, device)
            X_val_vec = embed_transformer(X_val_texts, tokenizer, emb_model, device)
            emb_size = X_tr_vec[0].shape[1]

        else:
            raise ValueError(f"Unsupported embedding method: {emb_method}")

        # Determine max sequence length for padding
        max_len = max(len(seq) for seq in X_tr_vec)

        # Pad sequences to max_len with zeros (post-padding)
        X_tr_pad = pad_sequences(X_tr_vec, maxlen=max_len, padding='post', dtype='float32')
        X_val_pad = pad_sequences(X_val_vec, maxlen=max_len, padding='post', dtype='float32')

        # One-hot encode labels (assuming 3 classes)
        y_tr_cat = tf.one_hot(y_tr, depth=3)
        y_val_cat = tf.one_hot(y_val, depth=3)

        # Create and compile the LSTM model
        model = create_lstm_model(input_length=max_len, emb_size=emb_size)

        # Train the model and validate on the validation fold
        history = model.fit(
            X_tr_pad,
            y_tr_cat,
            epochs=2,
            batch_size=32,
            verbose=1,
            validation_data=(X_val_pad, y_val_cat)
        )

        # Predict on validation data
        y_pred_probs = model.predict(X_val_pad)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Collect true and predicted labels (convert one-hot back to class indices)
        y_true_all.extend(np.argmax(y_val_cat, axis=1))
        y_pred_all.extend(y_pred)
        history_all.append(history)

    return y_true_all, y_pred_all, history_all


def get_label_map(transformer_name: str) -> Dict[str, int]:
    """
    Return a dictionary mapping model-specific output labels to integer class indices.

    This function helps translate string labels from different transformer models
    to consistent numeric labels used in downstream processing.

    Parameters
    ----------
    transformer_name : str
        Name or identifier of the transformer model.

    Returns
    -------
    Dict[str, int]
        A mapping from the model's output label strings to integer class indices.

    Raises
    ------
    ValueError
        If the transformer_name is not recognized or label mapping is unknown.
    """
    name = transformer_name.lower()

    if "yiyanghkust/finbert-tone" in name:
        # FinBERT label mapping
        return {
            "Negative": 0,  # Bearish
            "Positive": 1,  # Bullish
            "Neutral": 2
        }
    elif "cardiffnlp/twitter-roberta-base-sentiment" in name:
        # RoBERTa label mapping
        return {
            "LABEL_0": 0,  # Bearish (Negative)
            "LABEL_2": 1,  # Bullish (Positive)
            "LABEL_1": 2   # Neutral
        }
    else:
        raise ValueError(f"Unknown label mapping for model: {transformer_name}")
    
    
def eval_transformer(
    transformer: str,
    objective: str,
    skf,
    X_train: Union[pd.Series, List[str]],
    y_train: Union[pd.Series, List[int]],
) -> Tuple[List[int], List[int]]:
    """
    Evaluate a Hugging Face transformer model for text classification using cross-validation.

    This function uses the Hugging Face pipeline to perform inference on validation folds
    generated by the provided cross-validator. It maps predicted string labels to integers
    using a label map specific to the transformer model.

    Parameters
    ----------
    transformer : str
        Hugging Face model name or path (e.g., 'yiyanghkust/finbert-tone').
    objective : str
        Task name for pipeline (e.g., 'text-classification').
    skf :
        Cross-validation splitter (e.g., StratifiedKFold).
    X_train : Union[pd.Series, List[str]]
        Input texts for training.
    y_train : Union[pd.Series, List[int]]
        Ground-truth labels corresponding to X_train.

    Returns
    -------
    Tuple[List[int], List[int]]
        Tuple of (true labels, predicted labels) aggregated over all validation folds.
    """
    label_map = get_label_map(transformer)

    # Initialize the transformer pipeline for inference
    transf_pipeline = pipeline(
        task=objective,
        model=transformer,
        tokenizer=transformer,
        batch_size=32,
        framework="pt",
        device_map="cuda",  # GPU if available
        truncation=True
    )

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    # Iterate through train/validation splits
    for _, val_idx in skf.split(X_train, y_train):
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Predict on validation data
        preds = transf_pipeline(X_val.tolist())

        # Map string labels to integers according to model-specific mapping
        y_pred = [label_map[pred['label']] for pred in preds]

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

    return y_true_all, y_pred_all


def eval_llm_model(
    model: str
    ,skf
    ,X_train
    ,y_train
    ,system_message: str
    ,device_map: str = "cuda"
    ,max_new_tokens: int = 3
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(
        model
        ,device_map=device_map
        ,torch_dtype="auto"
        ,trust_remote_code=False
    )
    pipe = pipeline(
        "text-generation"
        ,model=lm
        ,tokenizer=tokenizer
        ,return_full_text=False
        ,max_new_tokens=max_new_tokens
        ,use_cache=True
    )

    def build_prompts(texts, system_message):
        return [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ]
            for text in texts
        ]

    def analyze_sentiment(prompt):
        outputs = pipe(prompt)
        answer = outputs[0]["generated_text"].strip()

        try:
            first_token = answer.split()[0]
            return first_token if first_token in {'0', '1', '2'} else 2 # Return 'Neutral' as default
        
        except ValueError:
            return 2 # Return 'Neutral' as default

    y_true_all = []
    y_pred_all = []

    for _, val_idx in skf.split(X_train, y_train):

        X_val = build_prompts(X_train.iloc[val_idx])
        y_val = y_train.iloc[val_idx]

        y_pred = X_val.apply(analyze_sentiment)

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

    return y_true_all, y_pred_all