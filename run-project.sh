#!/bin/bash
imagesFolder=./images
kernelType=$1
threadNum=$2
processNum=$3

g++ process-image-secuential.cpp -o process-image-secuential.exe
g++ -fopenmp process-image-omp.cpp -o process-image-omp.exe
nvcc process-image-cuda.cu -o process-image-cuda.exe
mpicc process-image-mpi.c -o process-image-mpi.exe

for image in "$imagesFolder"/*
do
    echo -e "\n"
    imageName="$(basename -s .jpg $image)"
    python3 convert-image-to-txt.py "${imageName}.jpg" $kernelType $threadNum
    ./process-image-secuential.exe ./results/txt_$imageName.txt ./results/data_$imageName.txt
    ./process-image-omp.exe ./results/txt_$imageName.txt ./results/data_$imageName.txt
    ./process-image-cuda.exe ./results/txt_$imageName.txt ./results/data_$imageName.txt
    mpirun -n $processNum --oversubscribe ./process-image-mpi.exe ./results/txt_$imageName.txt ./results/data_$imageName.txt
    convert ./results/${imageName}_sec_result.pgm ./results/${imageName}_sec_result.jpg
    convert ./results/${imageName}_omp_result.pgm ./results/${imageName}_omp_result.jpg
    convert ./results/${imageName}_cuda_result.pgm ./results/${imageName}_cuda_result.jpg
    convert ./results/${imageName}_omp_result.pgm ./results/${imageName}_mpi_result.jpg
    rm ./results/${imageName}_sec_result.pgm
    rm ./results/${imageName}_omp_result.pgm 
    rm ./results/${imageName}_cuda_result.pgm
    rm ./results/${imageName}_mpi_result.pgm
done