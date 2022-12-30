#include <iostream>
#include <fstream>
#include <cstring>
#include <omp.h>
using namespace std;

// Kernels (1 - 9)
int BLUR[3][3] = {{625, 1250, 625}, {1250, 2500, 1250}, {625, 1250, 625}};
int BOTTOM_SOBEL[3][3] = {{-1, -2, -1}, {0, 0, 0,}, {1, 2, 1}};
int EMBOSS[3][3] = {{-2, -1, 0}, {-1, 1, 1,}, {0, 1, 2}};
int IDENTITY[3][3] = {{0, 0, 0}, {0, 1, 0,}, {0, 0, 0}};
int LEFT_SOBEL[3][3] = {{1, 0, -1}, {2, 0, -2,}, {1, 0, -1}};
int OUTLINE[3][3] = {{-1, -1, -1}, {-1, 8, -1,}, {-1, -1, -1}};
int RIGHT_SOBEL[3][3] = {{-1, 0, 1}, {-2, 0, 2,}, {-1, 0, 1}};
int SHARPEN[3][3] = {{0, -1, 0}, {-1, 5, -1,}, {0, -1, 0}};
int TOP_SOBEL[3][3] = {{1, 2, 1}, {0, 0, 0,}, {-1, -2, -1}};

// Function to apply a kernel to an image
void applyKernel(int [], int [], int [][3], int, int, int, int);

// Main function
int main(int argc, char *argv[]) {
	// Variables
	string imageName;			// Image name without extension
	string imageExtention;		// Image extension 
	int width; 					// Width of the image
	int height;					// Height of the image
	int kernelNumber;			// Kernel number to apply
	int threadNumber;			// Number of threads
	int arraySize;				// Size of the image array
	int *imageMatrix;			// Image array
	int *resultMatrix;			// Result array

	// Read console arguments
	string imageTxt = argv[1];	// File name with the image in txt format
	string imageData = argv[2];	// File name with the image data

	// Opening the file
	ifstream imageDataFile(imageData);

	// Verify if file is open
	if(!imageDataFile.is_open())
		cout << "Error opening file";

	// Defining the loop for getting input from the file
	imageDataFile >> imageName;
	imageDataFile >> imageExtention;
	imageDataFile >> width;
	imageDataFile >> height;
	imageDataFile >> kernelNumber;
	imageDataFile >> threadNumber;

	arraySize = width * height;

	// Closing the file
	imageDataFile.close();

	// Allocating memory for the image array
	imageMatrix = (int *)malloc(arraySize * sizeof(int));
	resultMatrix = (int *)malloc(arraySize * sizeof(int));
	
	// Opening the file
	ifstream imageTxtFile(imageTxt);

	// Verify if file is open
	if(!imageTxtFile.is_open())
		cout << "Error opening file";

	// Defining the loop for getting input from the file
	for (int r = 0; r < arraySize; r++)
	{	
		imageTxtFile >> imageMatrix[r];
	}

	// Closing the file
	imageTxtFile.close();

	// Applying the kernel
	switch (kernelNumber)
	{
	case 1:
		applyKernel(imageMatrix, resultMatrix, BLUR, width, height, threadNumber, 10000);
		break;
	case 2:
		applyKernel(imageMatrix, resultMatrix, BOTTOM_SOBEL, width, height, threadNumber, 1);
		break;
	case 3:
		applyKernel(imageMatrix, resultMatrix, EMBOSS, width, height, threadNumber, 1);
		break;
	case 4:
		applyKernel(imageMatrix, resultMatrix, IDENTITY, width, height, threadNumber, 1);
		break;
	case 5:
		applyKernel(imageMatrix, resultMatrix, LEFT_SOBEL, width, height, threadNumber, 1);
		break;
	case 6:
			applyKernel(imageMatrix, resultMatrix, OUTLINE, width, height, threadNumber, 1);
		break;
	case 7:
		applyKernel(imageMatrix, resultMatrix, RIGHT_SOBEL, width, height, threadNumber, 1);
		break;
	case 8:
		applyKernel(imageMatrix, resultMatrix, SHARPEN, width, height, threadNumber, 1);
		break;
	case 9:
		applyKernel(imageMatrix, resultMatrix, TOP_SOBEL, width, height, threadNumber, 1);
		break;
	default:
		break;
	}

	
	//int (*)[3] kernel = SHARPEN;
	
	// Saving the result
	string fileDir = "./results/" + imageName + "_result.pgm"; 		// File path with the image in pgm format
	string fileName = "#" + imageName + "_result.pgm";				// File name with the image in pgm format
	string fileSize = to_string(width) + " " + to_string(height);	// File size with the image in pgm format

	// Opening the file
	ofstream imageResultFile(fileDir);
	imageResultFile << "P2";
	imageResultFile << "\n";
	imageResultFile << fileName;
	imageResultFile << "\n";
	imageResultFile << fileSize;
  	imageResultFile <<"\n";
	imageResultFile << "255";
	imageResultFile << "\n";

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			imageResultFile << resultMatrix[j + width * i];
			imageResultFile << " ";
		}
		imageResultFile << "\n";
	}
	imageResultFile.close();
}

void applyKernel(int image[],int result_image[], int kernel[][3], int width, int height, int threads, int factor) {
	int prod;
	
	#pragma omp parallel for set_num_thread(threads) firstprivate(prod)
	{
		for (int i = 0; i < width * height; i++) {
			if( (i < width) || (i % width == 0) || ( (i + 1) % width == 0) || (i > width * (height - 1))) {
				result_image[i] = 0;
			}
			else {
				prod = (kernel[0][0] * image[i - width - 1] +
					kernel[0][1] * image[i - width] +
					kernel[0][2] * image[i - width + 1] +
					kernel[1][0] * image[i - 1] +
					kernel[1][1] * image[i] +
					kernel[1][2] * image[i + 1] +
					kernel[2][0] * image[i + width - 1] +
					kernel[2][1] * image[i + width] +
					kernel[2][2] * image[i + width + 1]) / factor;
				if (prod < 0)
					prod = 0;
				if (prod > 255)
					prod = 255;
				result_image[i] = prod;
			}	
		}
	}
}
