import os
from moviepy.editor import *

input_folder = '/home/paul/zzz_drug_signals/data/mp4'
output_folder = '/home/paul/zzz_drug_signals/data/file_to_be_cut'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Loop through all files and process the mp4 files
for file in files:
    if file.endswith('.mp4'):
        file_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace('.mp4', '.wav'))

        # Load the video file
        video = VideoFileClip(file_path)

        # Extract the audio and set the sample rate
        audio = video.audio.set_fps(44100)

        # Save the audio as a wav file
        audio.write_audiofile(output_path, codec='pcm_s16le')
print("done!")
