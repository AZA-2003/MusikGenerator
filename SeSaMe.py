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
from dataloader import AudioLoader


class SeSaMe(nn.Module):
    def __init__(self,H=1,L=8000):
        super().__init__()
        self.l1 = nn.Linear(4*H, 2*H)
        self.l2 = nn.Linear(8*H, 4*H)
        self.l3 = nn.Linear(4*H, 8*H)
        self.l4 = nn.Linear(2*H, 4*H)
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
        
        #self.l2 = nn.Linear
    
    def forward(self,x):
        #print(x.shape)
        # x = x.view(x.shape[0], x.shape[1]//8, 8)
        x0 = x.view(x.shape[0], x.shape[1]//4, 4)
        #print(x0.shape)
        x1 = self.relu(self.l1(x0)) 
        print(x1.shape)
        x2 = self.relu(self.l2(x1.reshape(x1.shape[0], x1.shape[1]//4, x1.shape[2]*4)))
        print(x2.shape)
        x3 = x2+self.relu(self.mamba1(x2.reshape(x2.shape[0], x2.shape[2], x2.shape[1])))
        print(x3.shape)
        x4 = self.relu(self.l3(x3)).reshape(x3.shape[0], x3.shape[2]*4, x3.shape[1]//4)
        print(x4.shape)
        x5 = self.relu(self.mamba2(x4+x1))
        print(x5.shape)
        x6 = self.relu(self.l4(x5)).reshape(x5.shape[0], x5.shape[2]*4, x5.shape[1]//4)
        print(x6.shape)
        x7 = self.relu(self.mamba3(x6+x))
        print(x7.shape)
        y = x7.reshape(x7.shape[0], x7.shape[2]//4, x5.shape[1]*4)
        #print(x1.shape)
        #x2 = self.relu(self.mamba(x1))
        #print(x2.shape)
        #x3 = self.relu(self.l2(x2+x1))
        #print(x3.shape)
        #x4 = (x3+self.relu(self.skip_mamba(x))).reshape(x3.shape[0], x3.shape[1]*4, x3.shape[2]//4)
        #print(x4.shape)
        # y = self.relu(self.mamba2(x4)).reshape(x4.shape[0], x4.shape[2]//16, x4.shape[1]*16)
        #print(type(y), type(x))
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
    
    def evaluate(self, loader, epoch, setting="Validation"):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for x in tqdm(loader, desc=f"Epoch {epoch}", colour="green"):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(x)
                loss = self.criterion(y,x) ## works only with floating values
                running_loss += loss.item()  
            print(f"[Epoch {epoch}] {setting} Loss: {running_loss/len(loader):.4f}") 
    
    def synthesize(self, x,sr=16000):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y = self.model(x)
            y = y[0,0,:]
            y = y.cpu().numpy()
            print(y.shape)
            sf.write("test1.wav",y,sr)
            #librosa.output.write_wav("test1.wav",y.cpu().numpy(),sr=sr)
            
# model = SeSaMe()
# model  = model.to("cuda")
# for i,p in enumerate(loader.train_loader):
#     y = model(p.to("cuda"))
#     break
# class Lightning_SeSaMe(L.LightningModule):
#     def __init__(self,model):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         m = model
#         self.model = m.to(device)
#         self.criterion = nn.NLLoss()
        
#     def training_step(self, x):
#         x = x.to("cuda")
#         y = self.model(x)
#         loss = self.criterion(x,y)
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

DATAROOT = "youtube_mix"
loader  = AudioLoader(DATAROOT)
model = SeSaMe()      

sesame_pipeline = Pipeline(model,1e-4)
sesame_pipeline.train(loader.train_loader, loader.val_loader, 50)
sesame_pipeline.evaluate(loader.test_loader,"_",setting="Test")
sesame_pipeline.synthesize(torch.randn(1,160000))
        
    
        
        