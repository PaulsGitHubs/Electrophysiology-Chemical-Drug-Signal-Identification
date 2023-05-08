import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.dropout2 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6400, 128)  # Update 6400 based on the output size after conv2 and pool2 layers
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

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
    return torch.tensor(data).float()

model_path = 'audio_cnn_model_pytorch.pth'
num_classes = 2  # Replace with the number of classes in your dataset
model = AudioCNN(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

predict_folder = 'path/to/predict/folder'
predict_data = load_wav_files(predict_folder)
predict_data = preprocess_data(predict_data)

with torch.no_grad():
    output = model(predict_data)
    probabilities = F.softmax(output, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)

print("Predictions:", predicted_labels.numpy())
