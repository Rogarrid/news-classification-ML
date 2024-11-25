from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Evaluate the model using the test data, print the classification report, and plot the confusion matrix
def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred)
    print(report)
    plot_confusion_matrix(y_test, y_pred, classes=['economy', 'health', 'sports', 'technology'])

# Plot the confusion matrix for the given true and predicted labels
def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()