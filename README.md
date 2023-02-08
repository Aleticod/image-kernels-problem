# **Parallel and distributed algorithms course**
## **Research members**
1. *CESPEDES VILCA, Angel*
2. *GODOY LACUTA, Cristian*
3. *PFOCCORI QUISPE, Alex Harvey*
4. *PHUYO HUAMAN, Edson Leonid*
5. *QUISPE CLEMENTE, Saman*
## **Image Kernels**
The work consists of using parallel algorithms designed in OpenMP, CUDA and MPI for the Image Kernels issue. For more information see https://setosa.io/ev/image-kernels/

To do this, you would have to perform a versus of time and speedUp between the
3 parallelization techniques.

| Kernel name   | Number        |
| :----         |    ---:       |
| blur          | 1             |
| bottom sobel  | 2             |
| emboss        | 3             |
| identity      | 4             |
| left sobel    | 5             |
| outline       | 6             |
| right sobel   | 7             |
| sharpen       | 8             |
| top sobel     | 9             |

## **Programming languages**
The programming languages used in the project will be C++, python and bash.

## **Install necessary packages**
We need python3 and matplotlib package for install matplotlib package run the next command for your distribution

#### *Ubuntu / Debian*
    sudo apt install python3-matplotlib
#### *Fedora*
    sudo dnf install python3-matplotlib
#### *Red Hat*
    sudo yum install python3-matplotlib
#### *Arch*
    sudo pacman -S python-matplotlib

## Second step
We need ImageMagick, for install ImageMagick run the next command for your distribution
#### *Ubuntu / Debina*
    sudo apt install imagemagick
#### *Fedora*
    sudo dnf install ImageMagick
#### *Red Hat*
    sudo yum install ImageMagick
#### *Arch*
    sudo pacman -S imagemagick

## Third step
Verify if your PC has a NVIDIA GPU device, this is required to use the last version of CUDA. Verify with the next commands.


    lspci | grep -i nvidia
    nvidia-smi
    nvcc --version

To download and install CUDA visit the oficial page NVIDIA
https://developer.nvidia.com/cuda-downloads

## **Clone the repository**
Clone the repository and enter inside folder image-kernels-problem and create two empty folders ***images*** and ***results*** for download images and save results images respectively and run the next command

    git clone https://github.com/Aleticod/image-kernels-problem
    cd image-kernels-problem
Create two empty folders

    mkdir images results
## **Download images**
Download images from the next links and save in images folder, for this run the next commands

***Download 1-lena.jpg image with size 256x256***

    wget -P images https://i.postimg.cc/Y2znHqbR/1-lena.jpg

***Download 2-cameraman.jpg image with size 320x320***

    wget -P images https://i.postimg.cc/k4mZMdwn/2-cameraman.jpg

***Download 3-couple.jpg image with size 512x512***

    wget -P images https://i.postimg.cc/nL3y2LJC/3-couple.jpg

***Download 4-male.jpg image with size 1024x1024***

    wget -P images https://i.postimg.cc/yxZ27XSZ/4-male.jpg

## **Run the project**
To execute the project we must execute the following command, this program will be execute with all images, especify kernel number, number of threads and number of process as follows

    bash run-project.sh <kernel-number> <thread-numbers> <process-numbers>

Example: For image lena.jpg, kernel 3 and 128 threads

    bash run-project.sh 3 128 4

## **Results**
The results for this project are inside the ***results*** folder and are the following:
|Description                    |Result                |
|:----                          | ----:                 |
|Image file in txt format       |*txt_<image_name>.txt*                  |
Image data file in txt format   |*data_<image_name>.txt*                  |
Image result executed sequentially in jpg format        |*<image_name>_sec_result.jpg*       |
Image result executed with OpenMP in jpg format        |*<image_name>_omp_result.jpg*       |
Image result executed with CUDA in jpg format        |*<image_name>_cuda_result.jpg*       |
Image result executed with OpenMPI in jpg format        |*<image_name>_mpi_result.jpg*       |