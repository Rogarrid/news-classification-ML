import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_data(df):
    df['processed_text'] = df['headline'].apply(preprocess_text)
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed_text'])
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    padded_sequences = pad_sequences(sequences, padding='post')
    df['processed_text'] = list(padded_sequences)
    return df, tokenizer

def split_data(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['category'], test_size=0.2, random_state=42)
    return np.array(list(X_train)), np.array(list(X_test)), np.array(y_train), np.array(y_test)