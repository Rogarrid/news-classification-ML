import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

# Train the model and return it
def train_model(X_train, y_train):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=16, input_length=X_train.shape[1]),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    return model

# Save the model to the given file path
def save_model(model, file_path):
    model.save(file_path)

# Load the model from the given file path
def load_model(file_path):
    return tf.keras.models.load_model(file_path)