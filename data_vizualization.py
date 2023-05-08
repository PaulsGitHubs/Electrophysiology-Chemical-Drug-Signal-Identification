import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def load_audio_files(path: str, label:str):

    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        signal, _ = os.path.splitext(filename)
        signal_id, utterance_number = signal.split("_nohash_")
        utterance_number = int(utterance_number)
    
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append([waveform, sample_rate, label, signal_id, utterance_number])
        
    return dataset
    
trainset_drug_noise = load_audio_files('/home/zzz_drug_signal/data/drug_noise', 'drug_noise')
trainset_normal_static_noise = load_audio_files('/home/zzz_drug_signal/data/normal_static_noise', 'normal_static_noise')

trainloader_drug_noise = torch.utils.data.DataLoader(trainset_drug_noise, batch_size=1,
                                            shuffle=True, num_workers=0)
trainloader_normal_static_noise = torch.utils.data.DataLoader(trainset_normal_static_noise, batch_size=1,
                                            shuffle=True, num_workers=0)

drug_noise_waveform = trainset_drug_noise[0][0]
drug_noise_sample_rate = trainset_drug_noise[0][1]
print(f'Drug Noise Waveform: {drug_noise_waveform}')
print(f'Drug Noise Sample Rate: {drug_noise_sample_rate}')
print(f'Drug Noise Label: {trainset_drug_noise[0][2]}')
print(f'Drug Noise ID: {trainset_drug_noise[0][3]} \n')

normal_static_noise_waveform = trainset_normal_static_noise[0][0]
normal_static_noise_sample_rate = trainset_normal_static_noise[0][1]
print(f'Normal Static Noise Waveform: {normal_static_noise_waveform}')
print(f'Normal Static Noise Sample Rate: {normal_static_noise_sample_rate}')
print(f'Normal Static Noise Label: {trainset_normal_static_noise[0][2]}')
print(f'Normal Static Noise ID: {trainset_normal_static_noise[0][3]}')

def show_waveform(waveform, sample_rate, label):
    print("Waveform: {}\nSample rate: {}\nLabels: {} \n".format(waveform, sample_rate, label))
    new_sample_rate = sample_rate/10
   
    # Resample applies to a single channel, we resample first channel here
    channel = 0
    waveform_transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))

    print("Shape of transformed waveform: {}\nSample rate: {}".format(waveform_transformed.size(), new_sample_rate))

    plt.figure()
    plt.plot(waveform_transformed[0,:].numpy())
    
show_waveform(drug_noise_waveform, drug_noise_sample_rate, 'drug_noise')

show_waveform(normal_static_noise_waveform, normal_static_noise_sample_rate, 'normal_static_noise')

def show_spectrogram(waveform_drug_noise, waveform_normal_static_noise):
    drug_noise_spectrogram = torchaudio.transforms.Spectrogram()(waveform_drug_noise)
    print("\nShape of drug noise spectrogram: {}".format(drug_noise_spectrogram.size()))
    
    normal_static_noise_spectrogram = torchaudio.transforms.Spectrogram()(waveform_normal_static_noise)
    print("Shape of normal static noise spectrogram: {}".format(normal_static_noise_spectrogram.size()))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Features of {}".format('drug_noise'))
    plt.imshow(drug_noise_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    
    plt.subplot(1, 2, 2)
    plt.title("Features of {}".format('normal_static_noise'))
    plt.imshow(normal_static_noise_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    
show_spectrogram(drug_noise_waveform, normal_static_noise_waveform)

def show_melspectrogram(waveform, sample_rate):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mel_spectrogram.size()))

    plt.figure()
    plt.imshow(mel_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    
show_melspectrogram(drug_noise_waveform, drug_noise_sample_rate)

def show_mfcc(waveform, sample_rate):
    mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mfcc_spectrogram.size()))

    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(mfcc_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    
    plt.figure()
    plt.plot(mfcc_spectrogram.log2()[0,:,:].numpy())
    plt.draw()
    
show_mfcc(normal_static_noise_waveform, normal_static_noise_sample_rate)

def create_spectrogram_images(trainloader, label_dir):
    # Make directory
    directory = f'./data/spectrograms/{label_dir}/'
    if(os.path.isdir(directory)):
        print("Data exists for", label_dir)
    else:
        os.makedirs(directory, mode=0o777, exist_ok=True)
        
        for i, data in enumerate(trainloader):

            waveform = data[0]
            sample_rate = data[1][0]
            label = data[2]
            ID = data[3]

            # Create transformed waveforms
            spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)     
            
            fig = plt.figure()
            plt.imsave(f'./data/spectrograms/{label_dir}/spec_img{i}.png', spectrogram_tensor[0].log2()[0,:,:].numpy(), cmap='viridis')

def create_mfcc_images(trainloader, label_dir):
    # Make directory
    os.makedirs(f'./data/mfcc_spectrograms/{label_dir}/', mode=0o777, exist_ok=True)
    
    for i, data in enumerate(trainloader):

        waveform = data[0]
        sample_rate = data[1][0]
        label = data[2]
        ID = data[3]
        
        mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)

        plt.figure()
        fig1 = plt.gcf()
        plt.imshow(mfcc_spectrogram[0].log2()[0,:,:].numpy(), cmap='viridis')
        plt.draw()
        fig1.savefig(f'./data/mfcc_spectrograms/{label_dir}/spec_img{i}.png', dpi=100)
 
        #spectorgram_train.append([spectrogram_tensor, label, sample_rate, ID])
        
create_spectrogram_images(trainloader_drug_noise, 'drug_noise')
create_spectrogram_images(trainloader_normal_static_noise, 'normal_static_noise')
create_mfcc_images(trainloader_drug_noise, 'drug_noise')
create_mfcc_images(trainloader_normal_static_noise, 'normal_static_noise')           
