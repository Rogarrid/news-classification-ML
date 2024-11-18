import nltk
import os
import numpy as np
from nltk.data import find
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_model, save_model
from src.model_evaluation import evaluate_model, plot_confusion_matrix

# Download the necessary NLTK data if it is not already downloaded
try:
    find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

try:
    find('tokenizers/punkt.zip')
except LookupError:
    nltk.download('punkt')

# Create the models directory if it does not exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the data, preprocess it, and split it into training and testing sets
df = load_data('data/raw/news_headlines.csv')
df, tokenizer = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df)

# Train the model, save it, and evaluate it
model = train_model(X_train, y_train)
save_model(model, 'models/news_classifier.h5')

model = load_model('models/news_classifier.h5')

evaluate_model(model, X_test, y_test)