# Algoritmos paralelos y distribuidos
# Image-kernels-problem
This project is a solution for the image-kernels-problem

## First step
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

## Third step
Clone the repository and enter into folder image-kernels-problem and create two empty folders images and results for download images and save results images respectively and run the next command

    git clone https://github.com/Aleticod/image-kernels-problem
    cd image-kernels-problem
Create two empty folders

    mkdir images results
## Fourth step
Download images from the next link and save in images folder, for this run the next command

Download lena.jpg image with length 256x256

    wget -P images https://i.postimg.cc/Y2znHqbR/lena.jpg

Download cameraman.jpg image with length 512x512

    wget -P images https://i.postimg.cc/sfcNVV3V/cameraman.jpg

Download cat.jpg image with length 914x610

    wget -P images https://i.postimg.cc/sXVj0gT8/cat.jpg

Run the project with the next command

    bash run-project.sh <image-name> <kernel-number> <thread-numbers>

Example: For image lena.jpg, kernel 3 and 100 threads

    bash run-project.sh lena 3 100
