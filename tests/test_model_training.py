import unittest
import numpy as np
from src.model_training import train_model, save_model, load_model
from src.data_preprocessing import preprocess_text, preprocess_data, split_data, load_data

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.df = load_data('data/raw/news_headlines.csv')
        self.df, self.tokenizer = preprocess_data(self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)

    def test_preprocess_text(self):
        text = "This is a sample headline!"
        processed_text = preprocess_text(text)
        self.assertIsInstance(processed_text, str)
        self.assertNotIn("This", processed_text)
        self.assertNotIn("is", processed_text)
        self.assertIn("sample", processed_text)
        self.assertIn("headline", processed_text)

    def test_split_data(self):
        self.assertEqual(len(self.X_train), len(self.y_train))
        self.assertEqual(len(self.X_test), len(self.y_test))
        self.assertGreater(len(self.X_train), len(self.X_test))

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_model_accuracy(self):
        model = train_model(self.X_train, self.y_train)
        loss, accuracy = model.evaluate(self.X_test, self.y_test)
        self.assertGreater(accuracy, 0.8)

    def test_save_and_load_model(self):
        model = train_model(self.X_train, self.y_train)
        save_model(model, 'models/test_model.h5')
        loaded_model = load_model('models/test_model.h5')
        self.assertIsNotNone(loaded_model)
        self.assertTrue(hasattr(loaded_model, 'predict'))
        loss, accuracy = loaded_model.evaluate(self.X_test, self.y_test)
        self.assertGreater(accuracy, 0.8)

if __name__ == '__main__':
    unittest.main()