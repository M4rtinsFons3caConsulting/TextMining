import re
from tqdm import tqdm

import pandas as pd
import numpy as np

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Masking

def preprocess(
    dataframe: pd.DataFrame
    ,col_name: str
    ,keep_url: bool
    ,lemmatize: bool
    ,stemmize: bool
):
    updates = []
    stop = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')

    for j in tqdm(dataframe[col_name]):

        text = j

        # LOWERCASE TEXT
        text = text.lower()

        # REPLACE TICKERS WITH SPECIAL TOKEN
        text = re.sub(r"\$[A-Z]{1,5}", "[TICKER]", text)
        if keep_url:
            # REPLACE URL WITH SPECIAL TOKEN
            text = re.sub(r"http\S+", "[URL]", text)
        else:
            # REMOVE URLS
            text = re.sub(r"http\S+", "", text)
        # REMOVE PUNCTUATION
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

        # REMOVE STOPWORDS
        text = " ".join([word for word in text.split() if word not in stop])

        # LEMMATIZE
        if lemmatize:
            text = " ".join(lemma.lemmatize(word) for word in text.split())

        #Stemming
        if stemmize:
            text = " ".join(stemmer.stem(word) for word in text.split())

        updates.append(text)

    dataframe[col_name] = updates

    return dataframe


def eval_sklearn_model(vectorizer, classifier, skf, X_train, y_train):
    pipeline = Pipeline([
        ('vectorizer', vectorizer)
        ,('classifier', classifier)
    ])

    y_true_all = []
    y_pred_all = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

    return y_true_all, y_pred_all


def create_lstm_model(input_length, emb_size):
    '''model input in the shape(number of words per doc, word embedding size)'''
    input_ = Input(shape=(input_length, emb_size))

    '''mask layer to avoid model from considering padding vectors'''
    mask_layer = Masking(mask_value=0)
    mask = mask_layer(input_)

    '''BiLSTM layer'''
    lstm = Bidirectional(LSTM(units=32))(mask)

    '''activation layer'''
    act = Dense(3, activation='softmax')(lstm)

    '''model input and output'''
    model = Model(input_, act)

    '''model loss function and evaluation metrics'''

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def corpus2vec(corpus, w2v):
    index_set = set(w2v.index_to_key)
    word_vec = w2v.get_vector
    return [
        [word_vec(word) for word in doc.split() if word in index_set]
        for doc in tqdm(corpus)
    ]


def eval_lstm_model(vectorizer, emb_size, skf, X_train, y_train):
    tf.random.set_seed(220)
    y_true_all = []
    y_pred_all = []
    history_all = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Tokenize
        X_tr_vec = corpus2vec(X_tr, vectorizer)
        X_val_vec = corpus2vec(X_val, vectorizer)

        # Pad
        train_len = []
        for i in X_tr_vec:
            train_len.append(len(i))
        X_tr_pad = pad_sequences(sequences=X_tr_vec, maxlen=max(train_len), padding='post', dtype='float64')
        X_val_pad = pad_sequences(sequences=X_val_vec, maxlen=max(train_len), padding='post', dtype='float64')

        # Convert labels to numpy arrays
        y_tr = tf.one_hot(y_tr, depth=3)
        y_val = tf.one_hot(y_val, depth=3)

        # Build model
        model = create_lstm_model(input_length=max(train_len), emb_size=emb_size)

        # Train model
        history = model.fit(X_tr_pad, y_tr, epochs=2, batch_size=32, verbose=1, validation_data=(X_val_pad, y_val), validation_batch_size=32)

        # Predict
        y_pred_prob = model.predict(X_val_pad)
        y_pred = np.argmax(y_pred_prob, axis=1)

        y_true_all.extend(np.argmax(y_val, axis=1))
        y_pred_all.extend(y_pred)

        history_all.append(history)

    return y_true_all, y_pred_all, history_all