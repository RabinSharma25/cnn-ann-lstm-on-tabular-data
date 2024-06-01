import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input

# Load the datasets
data_path = 'data.csv'
labels_path = 'labels.csv'

# Read the data
data = pd.read_csv(data_path)
labels = pd.read_csv(labels_path)

# Convert data and labels to numpy arrays
data_np = data.values
labels_np = labels.values

# Reshape data for CNN: (number of samples, height, width, channels)
# Here, height=1, width=number of features, channels=1
data_reshaped = data_np.reshape((data_np.shape[0], data_np.shape[1], 1))

# One-hot encode the labels
labels_encoded = to_categorical(labels_np)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_reshaped, labels_encoded, test_size=0.20, random_state=42)

# Define the CNN model
model = Sequential([
    Input(shape=(data_np.shape[1], 1)),
    Conv1D(filters=64, kernel_size=1, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(filters=128, kernel_size=1, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
