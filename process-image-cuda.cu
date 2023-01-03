#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
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
__global__ void applyKernel(int *image,int *result_image, int kernel[][3], int *width, int *height, int *factor) {
	
	int prod;
	// ID calculation
	int idBlock = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int idThread = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	int id = idBlock * blockDim.x * blockDim.y * blockDim.z + idThread;

	// Calculating the image size
	int imageSize = *width * *height;

	// Applying the kernel
	if(id < imageSize) {
		if( (id < *width) || (id % *width == 0) || ( (id + 1) % *width == 0) || (id > *width * (*height - 1))) {
			result_image[id] = 0;
		}
		else {
			prod = (kernel[0][0] * image[id - *width - 1] +
				kernel[0][1] * image[id - *width] +
				kernel[0][2] * image[id - *width + 1] +
				kernel[1][0] * image[id - 1] +
				kernel[1][1] * image[id] +
				kernel[1][2] * image[id + 1] +
				kernel[2][0] * image[id + *width - 1] +
				kernel[2][1] * image[id + *width] +
				kernel[2][2] * image[id + *width + 1]) / *factor;
			if (prod < 0)
				prod = 0;
			if (prod > 255)
				prod = 255;
			result_image[id] = prod;
		}
	}
}

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
	int *dev_imageMatrix;		// Device image array
	int *dev_resultMatrix;		// Device result array

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
	
	// Allocating memory for the device image array
	cudaMalloc((void **)&dev_imageMatrix, arraySize * sizeof(int));
	cudaMalloc((void **)&dev_resultMatrix, arraySize * sizeof(int));

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

	// Copying the image array to the device
	cudaMemcpy(dev_imageMatrix, imageMatrix, arraySize * sizeof(int), cudaMemcpyHostToDevice);


	// Applying the kernel
	dim3 block(50, 50, 1);
	dim3 thread(50, 50, 1);

	switch (kernelNumber)
	{
	case 1:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, BLUR, &width, &height, 10000);
		break;
	case 2:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, BOTTOM_SOBEL, &width, &height, 1);
		break;
	case 3:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, EDGE_DETECT, &width, &height, 1);
		break;
	case 4:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, EMBOSS, &width, &height, 1);
		break;
	case 5:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, LEFT_SOBEL, &width, &height, 1);
		break;
	case 6:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, MEAN_REMOVAL, &width, &height, 1);
		break;
	case 7:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, RIGHT_SOBEL, &width, &height, 1);t
		break;
	case 8:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, SHARPEN, &width, &height, 1);
		break;
	case 9:
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, TOP_SOBEL, &width, &height, 1);
		break;
	default:
		break;
	}

	// Copying the result array to the host
	cudaMemcpy(resultMatrix, dev_resultMatrix, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

	// Saving the result
	string fileDir = "./results/" + imageName + "_omp_result.pgm"; 		// File path with the image in pgm format
	string fileName = "#" + imageName + "_omp_result.pgm";				// File name with the image in pgm format
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