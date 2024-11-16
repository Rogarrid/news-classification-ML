import unittest
import numpy as np
from src.model_training import train_model
from src.data_preprocessing import preprocess_text, preprocess_data, split_data, load_data

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.df = load_data('data/raw/news_headlines.csv')
        self.df, self.tokenizer = preprocess_data(self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_model_accuracy(self):
        model = train_model(self.X_train, self.y_train)
        loss, accuracy = model.evaluate(self.X_test, self.y_test)
        self.assertGreater(accuracy, 0.8) 

if __name__ == '__main__':
    unittest.main()