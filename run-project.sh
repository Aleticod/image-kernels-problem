#!/bin/bash
imageName=$1
kernelType=$2
threadNum=$3
python3 convert-image-to-txt.py "${imageName}.jpg" $kernelType $threadNum
g++ process-image-secuential.cpp -o process-image-secuential.exe
g++ -fopenmp process-image-omp.cpp -o process-image-omp.exe
./process-image-secuential.exe ./results/txt_$imageName.txt ./results/data_$imageName.txt
./process-image-omp.exe ./results/txt_$imageName.txt ./results/data_$imageName.txt
convert ./results/${imageName}_sec_result.pgm ./results/${imageName}_sec_result.jpg
convert ./results/${imageName}_omp_result.pgm ./results/${imageName}_omp_result.jpg
