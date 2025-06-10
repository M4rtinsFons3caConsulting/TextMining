import re
from tqdm import tqdm

import numpy as np

import string

from sklearn.pipeline import Pipeline

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Masking


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U00002B50"
        "\U00002B06"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def preprocess(
    text: str
    ,keep_ticker: bool
    ,keep_url: bool
    ,stopwords: set
    ,lemmatizer = None
    ,stemmizer = None
) -> str:
    if not isinstance(text, str):
        return ""
    
    black_list = ['eu', 'us', 'uk']

    # LOWERCASE TEXT
    text = text.lower()

    if keep_ticker:
        # REPLACE TICKERS WITH SPECIAL TOKEN
        text = re.sub(r"\$[a-zA-Z]+", "[TICKER]", text)
    else:
        # REMOVE TICKER
        text = re.sub(r"\$[a-zA-Z]+", "", text)

    if keep_url:
        # REPLACE URL WITH SPECIAL TOKEN
        text = re.sub(r"http\S+", "[URL]", text)
    else:
        # REMOVE URLS
        text = re.sub(r"http\S+", "", text)

    # REMOVE PRICES
    text = re.sub(r"[€$£¥]\d+(\.\d+)?", "", text)

    # REMOVE PUNCTUATION
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.replace("…", "")

    # REMOVE EMOJIS
    text = remove_emojis(text)

    # REMOVE OTHER SPECIAL CHARACTERS
    for char in ["’", "‘", '”', '“', "®", "™", "\x80", "→", "–", "•"]:
        text = text.replace(char, "")
    for char in ["\x8f", "\x9d", "—"]:
        text = text.replace(char, " ")

    # REMOVE STOPWORDS
    text = " ".join([word for word in text.split() if word not in stopwords])

    # LEMMATIZE
    if lemmatizer:
        text = " ".join(
            word if word in black_list
            else lemmatizer.lemmatize(word) 
            for word in text.split()
        )

    # STEMMIZE
    if stemmizer:
        text = " ".join(
            word if word in black_list
            else stemmizer.stem(word)
            for word in text.split()
        )

    return text


def find_punctuated_tokens(texts):
    punctuated = set()
    for text in texts:
        tokens = text.split()

        for token in tokens:
            if re.search(r'[^\w\s]', token):
                if '‑' in token:
                    pass
                else:
                    punctuated.add(token)
                    
    return punctuated


def eval_sklearn_model(
    vectorizer
    ,classifier
    ,skf
    ,X_train
    ,y_train
):
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


def generate_cls_embeddings(
    texts
    ,embeddings_model
    ,desc="Generating CLS Embeddings"
):
    """
    Generates CLS token embeddings for a list of input texts.

    Args:
        texts (list of str): List of text inputs.
        embeddings_model (callable): A Hugging Face embedding pipeline or model that returns hidden states.
        desc (str): Description for the tqdm progress bar.

    Returns:
        List of torch.Tensor: CLS embeddings for each input text.
    """
    cls_embeddings = []
    for text in tqdm(texts, desc=desc):
        embeddings = embeddings_model(text)
        # Assuming output format is [ [ [CLS], token1, token2, ... ] ]
        cls_embedding = torch.tensor(embeddings[0][0])
        cls_embeddings.append(cls_embedding)
    return cls_embeddings


def eval_transformer(
    transformer: str
    ,objective: str
    ,skf
    ,X_train
    ,y_train
    ,classifier=None
):
    transf_pipeline = pipeline(
        objective
        ,model=transformer
        ,tokenizer=transformer
        ,batch_size=32
        ,framework="pt"
        ,device_map="cuda"
        ,truncation=True
    )

    y_true_all = []
    y_pred_all = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        if classifier:
            X_tr = np.array(
                generate_cls_embeddings(
                    X_tr
                    ,embeddings_model=transf_pipeline
                )
            )
            y_tr = np.array(y_tr)

            X_val = np.array(
                generate_cls_embeddings(
                    X_val
                    ,embeddings_model=transf_pipeline
                )
            )
            y_val = np.array(y_val)

            classifier.fit(X_tr, y_tr)
            y_pred = classifier.predict(X_val)

        else:
            preds = transf_pipeline(X_val.tolist())
            y_pred = [
                2 if pred['label'] == 'LABEL_2'
                else 1 if pred['label'] == 'LABEL_1'
                else 0 for pred in preds
            ]

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

    def analyze_sentiment(query):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]

        outputs = pipe(messages)
        answer = outputs[0]["generated_text"].strip()

        first_token = answer.split()[0] if len(answer.split()) > 0 else None
        
        return first_token if first_token in {'0', '1', '2'} else '2'  # Return 'Neutral' as default

    y_true_all = []
    y_pred_all = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        y_pred = [analyze_sentiment(text) for text in X_val]

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

    return y_true_all, y_pred_all