import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_np)

# One-hot encode the labels
labels_encoded = to_categorical(labels_np)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels_encoded, test_size=0.2, random_state=42)

# Define the ANN model
model = Sequential([
    Dense(64, input_dim=data_scaled.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Assuming two classes: 0 and 1
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
