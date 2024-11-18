import unittest
from src.data_preprocessing import preprocess_text

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_text(self):
        text = "This is a sample headline!"
        processed_text = preprocess_text(text)
        self.assertIsInstance(processed_text, str)
        self.assertNotIn("This", processed_text)
        self.assertNotIn("is", processed_text)
        self.assertIn("sample", processed_text)
        self.assertIn("headline", processed_text)

    def test_remove_stopwords(self):
        text = "This is a test"
        processed_text = preprocess_text(text)
        self.assertNotIn("This", processed_text)
        self.assertNotIn("is", processed_text)
        self.assertIn("test", processed_text)

    def test_remove_punctuation(self):
        text = "Hello, world!"
        processed_text = preprocess_text(text)
        self.assertNotIn(",", processed_text)
        self.assertNotIn("!", processed_text)
        self.assertIn("hello", processed_text)
        self.assertIn("world", processed_text)

    def test_empty_text(self):
        text = ""
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "")

    def test_text_with_only_stopwords(self):
        text = "is the and"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "")

if __name__ == '__main__':
    unittest.main()