import nltk
from nltk.data import find
import os
from tensorflow.keras.models import load_model
import numpy as np

try:
    find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

try:
    find('tokenizers/punkt.zip')
except LookupError:
    nltk.download('punkt')

from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_model, save_model
from src.model_evaluation import evaluate_model, plot_confusion_matrix  # Importar plot_confusion_matrix

if not os.path.exists('models'):
    os.makedirs('models')

df = load_data('data/raw/news_headlines.csv')
df, tokenizer = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df)

model = train_model(X_train, y_train)
save_model(model, 'models/news_classifier.h5')

model = load_model('models/news_classifier.h5')

evaluate_model(model, X_test, y_test)

def show_predictions(model, X_test, y_test, tokenizer, num_examples=5):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    for i in range(num_examples):
        print(f"Texto: {tokenizer.sequences_to_texts([X_test[i]])}")
        print(f"Etiqueta verdadera: {y_test[i]}, Predicción: {y_pred[i]}")
        print()

show_predictions(model, X_test, y_test, tokenizer)

plot_confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=1), classes=['economy', 'health', 'sports', 'technology'])