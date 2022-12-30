#!/bin/bash
imageName=$1
kernelType=$2
threadNum=$3
python3 convert-image-to-txt.py "${imageName}.jpg" $kernelType $threadNum
g++ process-image.cpp -o process-image.exe
./process-image.exe ./results/txt_$imageName.txt ./results/data_$imageName.txt
convert ./results/${imageName}_result.pgm ./results/${imageName}_result.jpg
