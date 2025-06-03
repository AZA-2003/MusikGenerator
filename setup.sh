#!/bin/bash
## Setting up the variables
REQ="requirements.txt"
ZIP="youtube_mix.zip"
DATA="/youtube_mix"
## Setting up the packages for the environment
echo "Setting up the environment"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if [[ -f "$REQ" ]];
then
    pip install -r requirements.txt
else
    echo "requirements.txt does not exist!"
fi
echo "Installing MAMBA"
pip install mamba-ssm --no-build-isolation
## Setting up the data
echo "Setting up the data"
if [[ -d "$DATA" ]]; then
    echo "data already there"
elif [[ -f "$ZIP" ]]; then
    unzip youtube_mix.zip
else
    echo "zip file does not exist"
fi
echo "DONE!"