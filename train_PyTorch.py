import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

train_data = np.stack(train_data)[..., np.newaxis]
validation_data = np.stack(validation_data)[..., np.newaxis]
test_data = np.stack(test_data)[..., np.newaxis]

train_data = torch.tensor(train_data).float()
train_labels = torch.tensor(train_labels).long()
validation_data = torch.tensor(validation_data).float()
validation_labels = torch.tensor(validation_labels).long()
test_data = torch.tensor(test_data).float()
test_labels = torch.tensor(test_labels).long()

# Define CNN model using PyTorch
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

num_classes = 2  # Replace with the number of classes in your dataset
model = AudioCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Create data loaders
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

validation_dataset = TensorDataset(validation_data, validation_labels)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Train the model
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    model.train()

    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        output = model(batch_data)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output, 1)
        train_correct += (predicted == batch_labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # Validation
    model.eval()
    validation_loss = 0
    validation_correct = 0
    with torch.no_grad():
        for batch_data, batch_labels in validation_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            output = model(batch_data)
            loss = criterion(output, batch_labels)

            validation_loss += loss.item()
            _, predicted = torch.max(output, 1)
            validation_correct += (predicted == batch_labels).sum().item()

    validation_loss /= len(validation_loader.dataset)
    validation_acc = validation_correct / len(validation_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.6f}, Accuracy: {train_acc:.4f}, Validation Loss: {validation_loss:.6f}, Validation Accuracy: {validation_acc:.4f}")

# Evaluate the model
model.eval()
test_loss = 0
test_correct = 0
with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        output = model(batch_data)
        loss = criterion(output, batch_labels)

        test_loss += loss.item()
        _, predicted = torch.max(output, 1)
        test_correct += (predicted == batch_labels).sum().item()

test_loss /= len(test_loader.dataset)
test_acc = test_correct / len(test_loader.dataset)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Save the model
torch.save(model.state_dict(), "audio_cnn_model_pytorch.pth")
