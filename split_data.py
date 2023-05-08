import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

data_folder = '/home/paul/zzz_drug_signals/data/fent'
train_folder = '/home/paul/zzz_drug_signals/data/train'
validation_folder = '/home/paul/zzz_drug_signals/data/validation'
test_folder = '/home/paul/zzz_drug_signals/data/test' 

# Create the output folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get a list of all the WAV files in the data folder
wav_files = [f for f in os.listdir(data_folder) if f.endswith('.wav')]

# Split the file list into training, validation, and test sets
train_files, temp_files = train_test_split(wav_files, test_size=0.3, random_state=42)
validation_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

# Function to copy files from the original folder to the destination folder
def copy_files(files, src_folder, dst_folder):
    for file in files:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dst_folder, file)
        shutil.copy(src_path, dst_path)

# Copy the WAV files to their respective folders
copy_files(train_files, data_folder, train_folder)
copy_files(validation_files, data_folder, validation_folder)
copy_files(test_files, data_folder, test_folder)

# Print the number of files in each set
print("Training set size:", len(train_files))
print("Validation set size:", len(validation_files))
print("Test set size:", len(test_files))
