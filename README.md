# News Classifier

This project implements a news headline classification model that categorizes headlines into one of four categories: **Economy**, **Health**, **Sports**, and **Technology**. It was developed to build a Machine Learning model from scratch, covering all stages of the workflow: data collection, preprocessing, training, evaluation, and testing.

## Features

- Classifies news headlines into four categories: Economy, Health, Sports, and Technology.
- Preprocessing pipeline includes tokenization and stopword removal.
- Model evaluation with classification report, precision, recall, F1-score, and confusion matrix.
- Support for training a new model or loading an existing one.
- Includes unit tests for various stages of the pipeline (data preprocessing, model training, and evaluation).

## Technologies Used

- **Python**: Main programming language for implementing the model.
- **TensorFlow & Keras**: For building and training the neural network model.
- **scikit-learn**: For model evaluation (classification report and confusion matrix) and data preprocessing (Label Encoding, train-test split).
- **NLTK**: For natural language processing (tokenization, stopword removal).
- **Pandas**: For data manipulation and handling CSV files.
- **NumPy**: For numerical computations and array handling.
- **Matplotlib**: For plotting the confusion matrix.

## Prerequisites for starting a local project

Before starting, ensure that Python 3.8 or later is installed on your system. Additionally, you need to download some resources for NLTK:

```
python3 -m nltk.downloader stopwords punkt
```

## Installation

1. Clone this repository:

```
https://github.com/Rogarrid/news-classification-ML.git
```

2. Navigate to the project directory:

```
cd news-classification-ML
```

3. To install the necessary dependencies, run the following command in your terminal:

```
pip install -r requirements.txt
```

This will install the following main dependencies:

- numpy
- pandas
- scikit-learn
- tensorflow

## Running the Project

To train or load the model, execute the main script (main.py). It will perform the following tasks:

- Load the data from data/raw/news_headlines.csv.
- Preprocess the data (stopwords removal, tokenization, etc.).
- Train a new model if one does not exist in models/.
- Evaluate the model using test data.
- Display a classification report and confusion matrix.

Run the script as follows:

```
python3 main.py
```

## Results

During the evaluation, the model generates a report with the following metrics:

- Accuracy (Precision)
- Completeness (Recall)
- F1-Score
- Confusion Matrix (visualised using Matplotlib)

Example of confusion matrix output:

```
                 Predicted
             Economy  Health  Sports  Technology
Economy        30       0       0         0
Health          0      32       0         0
Sports          0       0      30         0
Technology      0       0       0        37
```

## Running Tests

The project includes unit tests to ensure code quality. These tests cover:

- Text preprocessing (test_data_preprocessing.py).
- Model training (test_model_training.py).
- Model evaluation (test_model_evaluation.py).

To run all the tests, use the following command:

```
python3 -m unittest discover tests
```

This will execute all the tests in the `tests/` folder.

## Project Structure

Here is the directory structure of the project:

```
news-classification-ML/
├── data/
│   ├── raw/
│       ├── news_headlines.csv
├── models/
│       ├── news_classifier.h5
│       ├── news_classifier.pkl
│       ├── test_model.h5
├── src/
│       ├── __init__.py
│       ├── data_preprocessing.py
│       ├── model_training.py
│       ├── model_evaluation.py
├── tests/
│       ├── test_data_preprocessing.py
│       ├── test_model_training.py
│       ├── test_model_evaluation.py
├── requirements.txt
├── main.py
├── README.md
└── .gitignore
```

## Contributions

If you want to improve this project, please open a **Pull Request** or create an **Issue** in this repository.

## Author

Rocio Garrido

Contact: rocio.garrido.fer@gmail.com
