import os
import numpy as np
import librosa

train_folder = '/home/paul/zzz_drug_signals/data/train'
validation_folder = '/home/paul/zzz_drug_signals/data/validation'
test_folder = '/home/paul/zzz_drug_signals/data/test'

data_directory = '/home/paul/zzz_drug_signals/data/npy_files'
os.makedirs(data_directory, exist_ok=True)

def load_and_save_data(folder, prefix):
    file_list = [f for f in os.listdir(folder) if f.endswith('.wav')]
    data = []

    for file in file_list:
        file_path = os.path.join(folder, file)
        audio, _ = librosa.load(file_path)
        data.append(audio)

    np.save(os.path.join(data_directory, f'{prefix}_data.npy'), data)

# Load and save data for each set
load_and_save_data(train_folder, 'train')
load_and_save_data(validation_folder, 'validation')
load_and_save_data(test_folder, 'test')
print("done!")
