## Pyotrch modules
import torch
from torch import nn
import torch.optim
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
import lightning as L
## basic libraries
import numpy as np
import matplotlib.pyplot as plt
## for display purposes
from tqdm.auto import trange, tqdm 
import IPython.display as ipd 
## for basic tensor operations
from einops import rearrange
## to read and display wav files
import glob
import librosa
import librosa.display
import soundfile as sf
## Mamba library
from mamba_ssm import Mamba
## custom dataloader for the data
from dataloader import AudioLoader, extract_waveform


class SeSaMe(nn.Module):
    def __init__(self,H=1,L=8000):
        super().__init__()
        ## Linear layers
        self.l1 = nn.Linear(4*H, 2*H)
        self.l2 = nn.Linear(8*H, 4*H)
        self.l3 = nn.Linear(4*H, 8*H)
        self.l4 = nn.Linear(2*H, 4*H)
        ## Mamba blocks
        self.mamba1 = Mamba(d_model= 4*H, ## model dimensionality (i.e dimension of the hidden state)
                           d_state=16,
                           d_conv=4,
                           expand=2
                           )
        self.mamba2 = Mamba(d_model= 2*H, ## model dimensionality (i.e dimension of the hidden state)
                           d_state=16,
                           d_conv=4,
                           expand=2
                           )
        self.mamba3 = Mamba(d_model= H, ## model dimensionality (i.e dimension of the hidden state)
                           d_state=16,
                           d_conv=4,
                           expand=2
                           )
        self.silu = nn.SiLU()
        self.relu = nn.GELU()

    '''
    forward pass of the model (includes skip connections between encoder and decoder)
    '''    
    def forward(self,x):
        x1 = x.reshape(x.shape[0], x.shape[1], 1)
        #print(x1.shape)
        x11 = x1.reshape(x1.shape[0], x1.shape[1]//4, x1.shape[2]*4)
        #print(x11.shape)
        x2 = self.relu(self.l1(x11))
        #print(x2.shape)
        x22 = x2.reshape(x2.shape[0], x2.shape[1]//4, x2.shape[2]*4)
        x3 = self.relu(self.l2(x22))
        # print(x3.shape)
        x4 = x3+self.relu(self.mamba1(x3))
        #print(x4.shape)
        x5 = self.relu(self.l3(x4))
        x55 = x5.reshape(x5.shape[0], x5.shape[1]*4, x5.shape[2]//4)
        x6 = self.relu(self.mamba2(x55+x2))
        #print(x6.shape)
        x7 = self.relu(self.l4(x6))
        x77 = x7.reshape(x7.shape[0], x7.shape[1]*4, x7.shape[2]//4)
        #print(x77.shape)
        y = self.relu(self.mamba3(x77+x1))
        y = y.reshape(y.shape[0], y.shape[1])
        return y
    
    '''
    encoder part of the model
    '''
    def encode(self,x):
        x1 = x.reshape(x.shape[0], x.shape[1], 1)
        #print(x1.shape)
        x11 = x1.reshape(x1.shape[0], x1.shape[1]//4, x1.shape[2]*4)
        #print(x11.shape)
        x2 = self.relu(self.l1(x11))
        #print(x2.shape)
        x22 = x2.reshape(x2.shape[0], x2.shape[1]//4, x2.shape[2]*4)
        x3 = self.relu(self.l2(x22))
        return x3
    '''
    decoder part of the model (w/out skip connections)
    '''
    def generate(self,x):
        x3 = self.relu(self.mamba1(x))
        x5 = self.relu(self.l3(x3))
        x55 = x5.reshape(x5.shape[0], x5.shape[1]*4, x5.shape[2]//4)
        x6 = self.relu(self.mamba2(x55))
        #print(x6.shape)
        x7 = self.relu(self.l4(x6))
        x77 = x7.reshape(x7.shape[0], x7.shape[1]*4, x7.shape[2]//4)
        #print(x77.shape)
        y = self.relu(self.mamba3(x77))
        y = y.reshape(y.shape[0], y.shape[1])
        return y


class Pipeline():
    def __init__(self,model, lr):
        self.device = torch.device("cuda") 
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader, val_loader, epochs):
        nn.init.xavier_uniform_(self.model.l1.weight)
        nn.init.normal_(self.model.l1.bias.data)
        nn.init.xavier_uniform_(self.model.l2.weight)
        nn.init.normal_(self.model.l2.bias.data)
        for e in range(epochs):
            self.model.train()
            running_loss = 0.0
            for x in tqdm(train_loader, desc=f"Epoch {e+1}", colour="yellow"):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(x)
                loss = self.criterion(y,x) ## works only with floating values
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()      
            self.evaluate(val_loader, e+1)
            print(f"[Epoch {e+1}] Training Loss: {running_loss/len(train_loader):.4f}")
        torch.save(self.model.state_dict(), "sesame_new_weights.pt")
    
    def evaluate(self, loader, epoch, setting="Validation"):
        self.model.eval()
        running_loss = 0.0
        batch_num = 0
        with torch.no_grad():
            for x in tqdm(loader, desc=f"Epoch {epoch}", colour="green"):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(x)
                loss = self.criterion(y,x) ## works only with floating values
                running_loss += loss.item()  
                if setting == "Test":
                    y = y[0,:]
                    y = y.cpu().numpy()
                    #print(y.shape)
                    sf.write(f"test{batch_num}.wav",y,16000)
                    batch_num+=1
            print(f"[Epoch {epoch}] {setting} Loss: {running_loss/len(loader):.4f}") 
    
    # def synthesize(self, x,sr=16000):
    #     self.model.eval()
    #     with torch.no_grad():
    #         x = x.to(self.device)
    #         pred = self.model.generate(x)
    #         y = pred[0,:]
    #         y = y.cpu().numpy()
    #         return pred,y
            

DATAROOT = "youtube_mix"
loader  = AudioLoader(DATAROOT)
model = SeSaMe()      

sesame_pipeline = Pipeline(model,1e-4)
#sesame_pipeline.train(loader.train_loader, loader.val_loader, 50)
sesame_pipeline.model.load_state_dict(torch.load("sesame_weights.pt", weights_only=True))
sesame_pipeline.evaluate(loader.test_loader,"_",setting="Test")
