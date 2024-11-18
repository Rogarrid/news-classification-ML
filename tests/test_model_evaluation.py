import unittest
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from src.model_evaluation import evaluate_model
from src.data_preprocessing import preprocess_data, split_data, load_data

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.df = load_data('data/raw/news_headlines.csv')
        self.df, self.tokenizer = preprocess_data(self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)

        self.model = Sequential([
            Embedding(input_dim=5000, output_dim=16, input_length=self.X_train.shape[1]),
            GlobalAveragePooling1D(),
            Dense(24, activation='relu'),
            Dense(4, activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, epochs=1, validation_split=0.2)

    def test_evaluate_model(self):
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        report = classification_report(self.y_test, y_pred)
        print(report)
        self.assertIsInstance(report, str)
        self.assertIn('accuracy', report)

if __name__ == '__main__':
    unittest.main()