## Pyotrch modules
import torch
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
## basic libraries
import numpy as np
import matplotlib.pyplot as plt
## for display purposes
from tqdm.auto import trange, tqdm 
import IPython.display as ipd 
## for basic tensor operations
from einops import rearrange
## to read and display wav files
import os
import glob ## just in case?
import random
import librosa
## for memmory management
import gc


SAMPLE_RATE = 16000
AUDIO_DURATION = 20 ## in seconds
MINI_DURATION = 5 ## in seconds
BATCH_SIZE = 8
TEST_BATCH_SIZE = 1
DATAROOT = "youtube_mix"

def extract_waveform(path):
    waveform, sr = librosa.load( DATAROOT+ '/' + path, sr=SAMPLE_RATE)
    waveform = np.array([waveform])
    if sr != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    # Pad so that everything is the right length
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = np.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_len]
    waveform = torch.FloatTensor(waveform).squeeze()
    return waveform

## for testing purposes
# print(extract_waveform("out000.wav").shape)

class AudioDataset(Dataset):
    '''
    class initialization
    '''
    def __init__(self, direc):
        self.meta = os.listdir(direc)
        self.data = []
        for p in tqdm(self.meta):
            wave  = extract_waveform(p)
            ## break down the waveform to smaller splits
            mini_len = SAMPLE_RATE * MINI_DURATION
            for i in range(int(AUDIO_DURATION/MINI_DURATION)):
                self.data.append(wave[i*mini_len:(i+1)*mini_len]) 
    '''
    get length of the dataset
    '''   
    def __len__(self):
        return len(self.data)

    '''
    get raw audio data--as a Tensor--given an index
    '''
    def __getitem__(self,idx):
        return self.data[idx]
    
## for testing purposes
# raw_audio = AudioDataset(DATAROOT)
# print(raw_audio.__len__())

class AudioLoader():
    
    def __init__(self,direc, split=0.86, seed=42):
        torch.manual_seed(seed)
        random.seed(seed)
        
        raw_audio = AudioDataset(direc)
        total_len = raw_audio.__len__()
        print(f"TOTAL LENGTH: {total_len}")
        
        train_len = int(split*total_len)
        val_len = int((total_len-train_len)/2)
        test_len = int(total_len-val_len-train_len)
        #split_len = [int(s*total_len) for s in split]
        split_len = [train_len,val_len, test_len]
        #print(sum(split_len), total_len)
        
        train_set,val_set, test_set = random_split(raw_audio, split_len)
        
        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        self.val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        self.test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE , shuffle=False, drop_last=True)  

## for testing purposes        
# loader  = AudioLoader(DATAROOT)
            