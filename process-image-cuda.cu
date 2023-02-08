#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
using namespace std;

// Kernels (1 - 9)
int kernelSize = 9;
int BLUR[9] = {625, 1250, 625, 1250, 2500, 1250, 625, 1250, 625};
int BOTTOM_SOBEL[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
int EMBOSS[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
int IDENTITY[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
int LEFT_SOBEL[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
int OUTLINE[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
int RIGHT_SOBEL[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
int SHARPEN[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
int TOP_SOBEL[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

// Function to apply a kernel to an image
__global__ void applyKernel(int *image,int *result_image, int *kernel, int width, int height, int factor) {
	
	int prod;
	// ID calculation
	int idBlock = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  	int idThread = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  	int id = blockDim.x * blockDim.y * blockDim.z * idBlock + idThread;

	// Calculating the image size
	int imageSize = width * height;

	// Applying the kernel
	if(id < imageSize) {
		if( (id < width) || (id % width == 0) || ( (id + 1) % width == 0) || (id > width * (height - 1))) {
			result_image[id] = 0;
		}
		else {
			prod = (kernel[0] * image[id - width - 1] +
				kernel[1] * image[id - width] +
				kernel[2] * image[id - width + 1] +
				kernel[3] * image[id - 1] +
				kernel[4] * image[id] +
				kernel[5] * image[id + 1] +
				kernel[6] * image[id + width - 1] +
				kernel[7] * image[id + width] +
				kernel[8] * image[id + width + 1]) / factor;
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
  	int *dev_kernel;			// Device kernel
	cudaEvent_t start;			// Time start
	cudaEvent_t end;			// Time end

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
  	cudaMalloc((void **)&dev_kernel, kernelSize * sizeof(int));

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
	dim3 block(25, 25, 5);
	dim3 thread(10, 10, 5);

	// Event creation
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Time start
	cudaEventRecord(start, 0);


	switch (kernelNumber)
	{
	case 1:
    cudaMemcpy(dev_kernel, BLUR, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 10000);
		break;
	case 2:
    cudaMemcpy(dev_kernel, BOTTOM_SOBEL, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 1);
		break;
	case 3:
    cudaMemcpy(dev_kernel, EMBOSS, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 1);
		break;
	case 4:
    cudaMemcpy(dev_kernel, IDENTITY, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 1);
		break;
	case 5:
    cudaMemcpy(dev_kernel, LEFT_SOBEL, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 1);
		break;
	case 6:
    cudaMemcpy(dev_kernel, OUTLINE, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 1);
		break;
	case 7:
    cudaMemcpy(dev_kernel, RIGHT_SOBEL, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 1);
		break;
	case 8:
    cudaMemcpy(dev_kernel, SHARPEN, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 1);
		break;
	case 9:
    cudaMemcpy(dev_kernel, TOP_SOBEL, kernelSize * sizeof(int), cudaMemcpyHostToDevice);
		applyKernel<<<block, thread>>>(dev_imageMatrix, dev_resultMatrix, dev_kernel, width, height, 1);
		break;
	default:
		break;
	}

	cudaDeviceSynchronize();

	// Time stop
	cudaEventRecord(end, 0);

	// Synchronization GPU - CPU
	cudaEventSynchronize(end);

	// Time calculation in milliseconds
	float time;
	cudaEventElapsedTime(&time, start, end);

	// Showing runtime results in seconds
	printf("> Runtime in CUDA: \t\t%f sec. time.\n", time / 1000);

	// Releasing resources
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	// Copying the result array to the host
	cudaMemcpy(resultMatrix, dev_resultMatrix, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

	// Saving the result
	string fileDir = "./results/" + imageName + "_cuda_result.pgm"; 		// File path with the image in pgm format
	string fileName = "#" + imageName + "_cuda_result.pgm";				// File name with the image in pgm format
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

  	cudaFree( dev_imageMatrix);
  	cudaFree( dev_kernel);
  	cudaFree( dev_resultMatrix);
}