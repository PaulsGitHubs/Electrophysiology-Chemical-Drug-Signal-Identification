import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the MFCC data and labels
data_directory = 'path/to/data/directory'
train_data = np.load(os.path.join(data_directory, 'train_mfcc.npy'))
validation_data = np.load(os.path.join(data_directory, 'validation_mfcc.npy'))
test_data = np.load(os.path.join(data_directory, 'test_mfcc.npy'))

# Load the labels for the audio files
# Replace this with the actual labels for your audio files
train_labels = np.random.randint(0, 2, len(train_data))
validation_labels = np.random.randint(0, 2, len(validation_data))
test_labels = np.random.randint(0, 2, len(test_data))

# Preprocess the MFCC data
train_data = np.stack(train_data)
validation_data = np.stack(validation_data)
test_data = np.stack(test_data)

# Add a channel dimension for compatibility with the CNN
train_data = train_data[..., np.newaxis]
validation_data = validation_data[..., np.newaxis]
test_data = test_data[..., np.newaxis]

# Define the CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=train_data.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Replace '2' with the number of classes in your dataset
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=20, validation_data=(validation_data, validation_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_labels)

# Print the test loss and accuracy
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Save the trained model
model.save("audio_cnn_model.h5")
