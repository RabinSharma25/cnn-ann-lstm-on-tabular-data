import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the datasets
data_path = 'data.csv'
labels_path = 'labels.csv'

# Read the data
data = pd.read_csv(data_path)
labels = pd.read_csv(labels_path)

# Convert data and labels to numpy arrays
data_np = data.values
labels_np = labels.values

# Reshape data for LSTM: (samples, time_steps, features)
# Here, each row is treated as a sequence with a single time step
data_reshaped = data_np.reshape((data_np.shape[0], data_np.shape[1], 1))

# One-hot encode the labels
labels_encoded = to_categorical(labels_np)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_reshaped, labels_encoded, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(50, input_shape=(data_np.shape[1], 1)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Assuming two classes: 0 and 1
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
