#!/bin/bash

# download BOLT-LMM
echo "downloading BOLT-LMM"
wget https://storage.googleapis.com/broad-alkesgroup-public/BOLT-LMM/downloads/BOLT-LMM_v2.3.4.tar.gz
tar -xvf BOLT-LMM_v2.3.4.tar.gz
rm BOLT-LMM_v2.3.4.tar.gz

# EyePACS Resnet50
echo "downloading pretrained ResNet50"
if [ ! -d models ]; then
    mkdir models
fi
wget https://www.dropbox.com/s/h5clve72fpwe3ku/resnet50_eyepacs.pt -O models/resnet50_eyepacs.pt

# StyleGAN2
echo "downloading pretrained StyleGAN2"
if [ ! -d models/stylegan2_healthy ]; then
    mkdir models/stylegan2_healthy/
fi
wget https://www.dropbox.com/s/ldsfejcnho3izok/model_40.pt -O models/stylegan2_healthy/model_40.pt

echo "done"
