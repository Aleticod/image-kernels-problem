#!/bin/bash
imageName=$1
kernelType=$2
python3 convert-image-to-txt.py "${imageName}.jpg" $kernelType
g++ process-image.cpp -o process-image.exe
./process-image.exe ./results/txt_$imageName.txt ./results/data_$imageName.txt $kernelType
convert ./results/${imageName}_result.pgm ./results/${imageName}_result.jpg
