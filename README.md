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

#### *Ubuntu / Debina*
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

## **Clon the repository**
Clone the repository and enter inside folder image-kernels-problem and create two empty folders ***images*** and ***results*** for download images and save results images respectively and run the next command

    git clone https://github.com/Aleticod/image-kernels-problem
    cd image-kernels-problem
Create two empty folders

    mkdir images results
## **Download images**
Download images from the next links and save in images folder, for this run the next commands

***Download lena.jpg image with size 256x256***

    wget -P images https://i.postimg.cc/Y2znHqbR/lena.jpg

***Download cameraman.jpg image with size 512x512***

    wget -P images https://i.postimg.cc/sfcNVV3V/cameraman.jpg

***Download cat.jpg image with size 914x610***

    wget -P images https://i.postimg.cc/sXVj0gT8/cat.jpg

## **Run the project**
To execute the project we must execute the following command, with the name of the image without extension, kernel number and number of threads as follows

    bash run-project.sh <image-name> <kernel-number> <thread-numbers>

Example: For image lena.jpg, kernel 3 and 1000 threads

    bash run-project.sh lena 3 1000

## **Results**
The results for this project are inside the ***results*** folder and are the following:
|Description                    |Result                |
|:----                          | ----:                 |
|Image file in txt format       |*txt_<image_name>.txt*                  |
Image data file in txt format   |*data_<image_name>.txt*                  |
Image result executed sequentially in pgm format        |*<image_name>_sec_result.pgm*      |
Image result executed with OpenMP in pgm format        |*<image_name>_omp_result.pgm*       |
Image result executed sequentially in jpg format        |*<image_name>_sec_result.jpg*       |
Image result executed with OpenMP in jpg format        |*<image_name>_omp_result.jpg*       |
