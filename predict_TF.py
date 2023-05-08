import os
import numpy as np
import librosa
import tensorflow as tf

def load_wav_files(folder):
    file_list = [f for f in os.listdir(folder) if f.endswith('.wav')]
    data = []

    for file in file_list:
        file_path = os.path.join(folder, file)
        audio, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(audio, sr=sr)
        data.append(mfcc)

    return data

def preprocess_data(data):
    data = np.stack(data)[..., np.newaxis]
    return data.astype(np.float32)

model_path = 'audio_cnn_model_tf.h5'
model = tf.keras.models.load_model(model_path)

predict_folder = 'path/to/predict/folder'
predict_data = load_wav_files(predict_folder)
predict_data = preprocess_data(predict_data)

predictions = model.predict(predict_data)
predicted_labels = np.argmax(predictions, axis=1)

print("Predictions:", predicted_labels)
