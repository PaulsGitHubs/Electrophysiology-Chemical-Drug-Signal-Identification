import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load and preprocess WAV files
def load_wav_files(folder):
    file_list = [f for f in os.listdir(folder) if f.endswith('.wav')]
    data = []

    for file in file_list:
        file_path = os.path.join(folder, file)
        audio, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(audio, sr=sr)
        data.append(mfcc)

    return data

data_folder = 'path/to/your/data/folder'
train_folder = 'path/to/train/folder'
validation_folder = 'path/to/validation/folder'
test_folder = 'path/to/test/folder'

train_data = load_wav_files(train_folder)
validation_data = load_wav_files(validation_folder)
test_data = load_wav_files(test_folder)

# Load labels and preprocess
train_labels = np.random.randint(0, 2, len(train_data))
validation_labels = np.random.randint(0, 2, len(validation_data))
test_labels = np.random.randint(0, 2, len(test_data))

train_data = np.stack(train_data)
validation_data = np.stack(validation_data)
test_data = np.stack(test_data)

train_data = train_data[..., np.newaxis]
validation_data = validation_data[..., np.newaxis]
test_data = test_data[..., np.newaxis]

# Define CNN model
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

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)

# Print test loss and accuracy
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Save the model
model.save("audio_cnn_model.h5")
