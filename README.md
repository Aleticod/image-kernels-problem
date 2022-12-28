# Algoritmos paralelos
# image-kernels-problem
This is a algorithm parallel project

## First step
We need to ensure in our pc have the matplotlib python library
### Ubuntu / Debina
    $ sudo apt install python3-matplotlib
### Fedora
    $ sudo dnf install python3-matplotlib
### Red Hat
    $ sudo yum install python3-matplotlib
### Arch
    $ sudo pacman -S python-matplotlib

## Second step
We need imagemagick package
### Ubuntu / Debina
    $ sudo apt imagemagick
### Fedora
    $ sudo dnf install ImageMagick
### Red Hat
    $ sudo yum install ImageMagick
### Arch
    $ sudo pacman -S imagemagick

## Third step
Clone the repository and enter the repository

    $ git clone https://github.com/Aleticod/image-kernels-problem
    $ cd image-kernels-problem
Create two empty folders

    $ mkdir images results
Download square gray image png into images folder and run the next command

    $ bash ./run-project.sh <image-name> <kernel-number>
Example: For image lenna.png and kernel 3

    $ bash ./run-project.sh lenna 3
