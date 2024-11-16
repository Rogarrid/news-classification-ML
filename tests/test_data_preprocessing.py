import unittest
from src.data_preprocessing import preprocess_text

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_text(self):
        text = "This is a sample headline!"
        processed_text = preprocess_text(text)
        self.assertIsInstance(processed_text, str)

if __name__ == '__main__':
    unittest.main()