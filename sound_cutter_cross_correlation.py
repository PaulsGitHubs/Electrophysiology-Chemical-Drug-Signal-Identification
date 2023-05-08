import os
import numpy as np
import librosa
from scipy.io.wavfile import write
'''
You can customize the threshold parameter to control the strictness of the similarity check. A higher threshold value will result in fewer, more similar cuts, while a lower threshold will produce more, less similar cuts.
'''
def find_similar_sounds(sound, target_sound, threshold=0.5):
    """
    Find parts of 'sound' that resemble 'target_sound' based on cross-correlation.

    Args:
        sound (np.array): The sound to search in.
        target_sound (np.array): The target sound to find.
        threshold (float): The similarity threshold. Higher values are more strict.

    Returns:
        List of tuples: Each tuple contains the start and end indices of similar sounds.
    """
    correlation = np.correlate(sound, target_sound, mode='valid')
    max_correlation = np.max(correlation)
    similarity = correlation / max_correlation

    indices = np.where(similarity >= threshold)[0]
    segments = []

    for idx in indices:
        start = idx
        end = idx + len(target_sound)
        segments.append((start, end))

    return segments

input_folder = '/home/paul/zzz_drug_signals/data/files_to_be_cut'
output_folder = '/home/paul/zzz_drug_signals/data/fent'
target_sound_file = '/home/paul/zzz_drug_signals/data/Sound_for_cutting/strong_dose_fent_crack.wav' #'/home/paul/zzz_drug_signals/data/Sound_for_cutting/saveusfent.wav' 
desired_sample_rate = 44100

# Load the target sound
target_sound, target_sample_rate = librosa.load(target_sound_file, sr=desired_sample_rate)
print("loaded target sound")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = os.listdir(input_folder)

for file in files:
    if file.endswith('.wav'):
        file_path = os.path.join(input_folder, file)
        output_base = os.path.join(output_folder, file.replace('.wav', ''))
        
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=desired_sample_rate)

        print("loaded sound_file to be cut")
        # Find parts that resemble the target sound
        segments = find_similar_sounds(audio, target_sound)

        # Cut the audio and save the identified parts
        for i, (start, end) in enumerate(segments):
            cut_audio = audio[start:end]
            print("cut one")
            output_path = f'{output_base}_cut_{i}.wav'
            write(output_path, desired_sample_rate, (cut_audio * 32767).astype(np.int16))
            print("wrote one")
