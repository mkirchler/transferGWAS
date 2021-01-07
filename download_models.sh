#!/bin/bash

# EyePACS Resnet50
if [ ! -d models ]; then
    mkdir models
fi
wget https://www.dropbox.com/s/h5clve72fpwe3ku/resnet50_eyepacs.pt -O models/resnet50_eyepacs.pt

# StyleGAN2
if [ ! -d models/stylegan2_healthy ]; then
    mkdir models/stylegan2_healthy/
fi
wget https://www.dropbox.com/s/ldsfejcnho3izok/model_40.pt -O models/stylegan2_healthy/model_40.pt
