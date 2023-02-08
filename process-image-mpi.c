#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

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

// Main function
int main(int argc, char *argv[]) {
	// Variables
	FILE *imageDataFile;
	FILE *imageTxtFile;
	char imageTxt[50];			// File name with the image in txt format
	char imageData[50];
	char imageName[50];			// Image name without extension
	char imageExtention[6];		// Image extension 
	int width; 					// Width of the image
	int height;					// Height of the image
	int kernelNumber;			// Kernel number to apply
	int threadNumber;			// Number of threads
	int arraySize;				// Size of the image array
	int *imageMatrix;			// Image array
	int *resultMatrix;
	int num;
	double start, end, time_local, time;
	int kernel[9];

	// Read console arguments
	strcpy(imageTxt, argv[1]);	// File name with the image in txt format
	strcpy(imageData, argv[2]); // File name with the image data

	// Opening the file
	imageDataFile = fopen(imageData, "r");

	// Verify if file is open
	if(imageDataFile == NULL)
		printf("Error al abrir el archivo\n");

	// Defining the loop for getting input from the file
	char width_s[5];
	char height_s[5];
	char kernelNumber_s[5];

	fscanf(imageDataFile, "%s", &imageName);
	fscanf(imageDataFile, "%s", &imageExtention);
	fscanf(imageDataFile, "%s", &width_s);
	fscanf(imageDataFile, "%s", &height_s);
	fscanf(imageDataFile, "%s", &kernelNumber_s);

	width = atoi(width_s);
	height = atoi(height_s);
	kernelNumber = atoi(kernelNumber_s);
	// Closing the file
	fclose(imageDataFile);

	arraySize = width * height;

	// Allocating memory for the image array
	imageMatrix = (int *)malloc(arraySize * sizeof(int));
	resultMatrix = (int *)malloc(arraySize * sizeof(int));
	
	// Opening the file
	imageTxtFile = fopen(imageTxt, "r");

	// Verify if file is open
	if(imageTxtFile == NULL)
		printf("Error\n");

	// Defining the loop for getting input from the file
	for (int r = 0; r < arraySize; r++)
	{	
		fscanf(imageTxtFile, "%d", &num);
		imageMatrix[r] = num;
	}

	// Closing the file
	fclose(imageTxtFile);

	int comm_sz;
	int my_rank;
	int dest;
	int factor;
	

	// Applying the kernel
	switch (kernelNumber)
	{
	case 1:
		factor = 10000;
		for (int j = 0; j < 9; j++) {
			kernel[j] = BLUR[j];
		}
		
		break;
	case 2:
		factor = 1;
		for (int j = 0; j < 9; j++) {
			kernel[j] = TOP_SOBEL[j];
		}
		break;
	case 3:
		factor = 1;
		for (int j = 0; j < 9; j++) {
			kernel[j] = EMBOSS[j];
		}
		break;
	case 4:
		factor = 1;
		for (int j = 0; j < 9; j++) {
			kernel[j] = IDENTITY[j];
		}
		break;
	case 5:
		factor = 1;
		for (int j = 0; j < 9; j++) {
			kernel[j] = LEFT_SOBEL[j];
		}
		break;
	case 6:
		factor = 1;
		for (int j = 0; j < 9; j++) {
			kernel[j] = OUTLINE[j];
		}
		break;
	case 7:
		factor = 1;
		for (int j = 0; j < 9; j++) {
			kernel[j] = RIGHT_SOBEL[j];
		}
		break;
	case 8:
		factor = 1;
		for (int j = 0; j < 9; j++) {
			kernel[j] = SHARPEN[j];
		}
		break;
	case 9:
		factor = 1;
		for (int j = 0; j < 9; j++) {
			kernel[j] = BOTTOM_SOBEL[j];
		}
		break;
	default:
		break;
	}

	// Initialization MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	// Initialize start time
	start=MPI_Wtime();

	// Send data to each process
	if (my_rank == 0) {	
		for (dest = 1; dest < comm_sz; dest++) {
			MPI_Send(imageMatrix, arraySize, MPI_INT, dest, 0, MPI_COMM_WORLD);
			MPI_Send(kernel, 9, MPI_INT, dest, 0, MPI_COMM_WORLD);
			MPI_Send(resultMatrix, arraySize, MPI_INT, dest, 0, MPI_COMM_WORLD);
		}
	}
	// Each process recive data from process 0
	else {
		MPI_Recv(imageMatrix, arraySize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(kernel, 9, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(resultMatrix, arraySize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Applying kernel for each process
	int prod;	
	int m = arraySize / comm_sz;
	for (int i = my_rank * m; i < (my_rank + 1) * m; i++) {
		if( (i < width) || (i % width == 0) || ( (i + 1) % width == 0) || (i > width * (height - 1))) {
			resultMatrix[i] = 0;
		}
		else {
			prod = (kernel[0] * imageMatrix[i - width - 1] +
				kernel[1] * imageMatrix[i - width] +
				kernel[2] * imageMatrix[i - width + 1] +
				kernel[3] * imageMatrix[i - 1] +
				kernel[4] * imageMatrix[i] +
				kernel[5] * imageMatrix[i + 1] +
				kernel[6] * imageMatrix[i + width - 1] +
				kernel[7] * imageMatrix[i + width] +
				kernel[8] * imageMatrix[i + width + 1]) / factor;
			if (prod < 0)
				prod = 0;
			if (prod > 255)
				prod = 255;
			resultMatrix[i] = prod;
		}
	}

	end = MPI_Wtime();

	// Calculate the time
	time_local = end - start;
	MPI_Reduce(&time_local,&time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	// Each process send data to proces 0
	if (my_rank != 0) {
		MPI_Send(resultMatrix, arraySize,MPI_INT,0,0,MPI_COMM_WORLD);
	}		
	// Process 0 recive results from other process	
	else {
		int q;
		for (q = 1; q < comm_sz; q++) {
			MPI_Recv(resultMatrix, arraySize, MPI_INT, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		// Process 0 apply kernel
		for (int i = my_rank * m; i < (my_rank + 1) * m; i++) {
			if( (i < width) || (i % width == 0) || ( (i + 1) % width == 0) || (i > width * (height - 1))) {
				resultMatrix[i] = 0;
			}
			else {
				prod = (kernel[0] * imageMatrix[i - width - 1] +
					kernel[1] * imageMatrix[i - width] +
					kernel[2] * imageMatrix[i - width + 1] +
					kernel[3] * imageMatrix[i - 1] +
					kernel[4] * imageMatrix[i] +
					kernel[5] * imageMatrix[i + 1] +
					kernel[6] * imageMatrix[i + width - 1] +
					kernel[7] * imageMatrix[i + width] +
					kernel[8] * imageMatrix[i + width + 1]) / factor;
				if (prod < 0)
					prod = 0;
				if (prod > 255)
					prod = 255;
				resultMatrix[i] = prod;
			}
		}
		printf("> Runtime in OpenMPI: \t\t%f sec. time.\n", time);
	}	
	
	// Finalize MPI 
	MPI_Finalize();
	// Saving the result

	FILE *imageResultFile;
	char fileDir[50];
	strcat(fileDir, "./results/" );
	strcat(fileDir, imageName );
	strcat(fileDir,"_mpi_result.pgm");
	char fileName[50] = "";
	strcat(fileName, "#" );
	strcat(fileName, imageName );
	strcat(fileName,"_mpi_result.pgm");
	char fileSize[50] = "";
	strcat(fileSize, width_s);
	strcat(fileSize, " ");
	strcat(fileSize, height_s);

	// Opening the file
	imageResultFile = fopen(fileDir, "w");
	fprintf(imageResultFile, "P2");
	fprintf(imageResultFile, "\n");
	fprintf(imageResultFile, fileName);
	fprintf(imageResultFile, "\n");
	fprintf(imageResultFile, fileSize);
	fprintf(imageResultFile, "\n");
	fprintf(imageResultFile, "255");
	fprintf(imageResultFile, "\n");

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			num = resultMatrix[j + width * i];
			char num_str[20];
			sprintf(num_str, "%d", num);
			fprintf(imageResultFile, num_str);
			fprintf(imageResultFile, " ");

		}
		fprintf(imageResultFile, "\n");
	}
	fclose(imageResultFile);
}

