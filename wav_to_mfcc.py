import os
import numpy as np
import librosa

train_folder = 'path/to/train/folder'
validation_folder = 'path/to/validation/folder'
test_folder = 'path/to/test/folder'

data_directory = 'path/to/data/directory'
os.makedirs(data_directory, exist_ok=True)

def wav_to_mfcc(folder, prefix):
    file_list = [f for f in os.listdir(folder) if f.endswith('.wav')]
    mfcc_data = []

    for file in file_list:
        file_path = os.path.join(folder, file)
        audio, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(audio, sr=sr)
        mfcc_data.append(mfcc)

    np.save(os.path.join(data_directory, f'{prefix}_mfcc.npy'), mfcc_data)

# Convert WAV files to MFCCs for each set
wav_to_mfcc(train_folder, 'train')
wav_to_mfcc(validation_folder, 'validation')
wav_to_mfcc(test_folder, 'test')
