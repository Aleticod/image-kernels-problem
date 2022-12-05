#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

// Kernels
double BLUR_KERNEL[3][3] = {{0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625}};
int BOTTOM_SOBEL[3][3] = {{-1, -2, -1}, {0, 0, 0,}, {1, 2, 1}};
int EMBOSS[3][3] = {{-2, -1, 0}, {-1, 1, 1,}, {0, 1, 2}};
int IDENTITY[3][3] = {{0, 0, 0}, {0, 1, 0,}, {0, 0, 0}};
int LEFT_SOBEL[3][3] = {{1, 0, -1}, {2, 0, -2,}, {1, 0, -1}};
int OUTLINE[3][3] = {{-1, -1, -1}, {-1, 8, -1,}, {-1, -1, -1}};
int RIGHT_SOBEL[3][3] = {{-1, 0, 1}, {-2, 0, 2,}, {-1, 0, 1}};
int SHARPEN[3][3] = {{0, -1, 0}, {-1, 5, -1,}, {0, -1, 0}};
int TOP_SOBEL[3][3] = {{1, 2, 1}, {0, 0, 0,}, {-1, -2, -1}};

void applyKernel(int [], int [], int [][3], int);

int main(int argc, char *argv[]) {
	// Read console arguments
	string imageName = argv[1];
	string dataInfo = argv[2];
	string name;
	string extention;
	int height;
	int width;
	int kernelNumber;

	// Opening the file
	ifstream datafile(dataInfo);

	// Verify if file is open
	if(!datafile.is_open())
		cout << "Error opening file";

	// Defining the loop for getting input from the file
	datafile >> name;
	datafile >> extention;
	datafile >> width;
	height = width;
	datafile >> kernelNumber;	

	int arraySize = width * height;

	// Lectura del archivo lena320.txt
	int image_matriz[arraySize];
	int result_matriz[arraySize];
	
	// Opening the file
	ifstream inputfile(imageName);

	// Verify if file is open
	if(!inputfile.is_open())
		cout << "Error opening file";

	// Defining the loop for getting input from the file
	for (int r = 0; r < arraySize; r++) {
		inputfile >> image_matriz[r];
	}

	switch (kernelNumber)
	{
	case 1/* constant-expression */:
		/* code */
		break;
	case 2:
		applyKernel(image_matriz, result_matriz, BOTTOM_SOBEL, width);
		break;
	case 3:
		applyKernel(image_matriz, result_matriz, EMBOSS, width);
		break;
	case 4:
		applyKernel(image_matriz, result_matriz, IDENTITY, width);
		break;
	case 5:
		applyKernel(image_matriz, result_matriz, LEFT_SOBEL, width);
		break;
	case 6:
			applyKernel(image_matriz, result_matriz, OUTLINE, width);
		break;
	case 7:
		applyKernel(image_matriz, result_matriz, RIGHT_SOBEL, width);
		break;
	case 8:
		applyKernel(image_matriz, result_matriz, SHARPEN, width);
		break;
	case 9:
		applyKernel(image_matriz, result_matriz, TOP_SOBEL, width);
		break;
	default:
		break;
	}

	
	//int (*)[3] kernel = SHARPEN;

	string fileSize = to_string(width) + " " + to_string(height);
	string fileDir = "./results/" + name + "_result.pgm";
	string fileName = "#" + name + "_result.pgm";

	ofstream imagen(fileDir);
	imagen << "P2";
	imagen << "\n";
	imagen << fileName;
	imagen << "\n";
	imagen << fileSize;
  	imagen <<"\n";
	imagen << "255";
	imagen << "\n";
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			imagen << result_matriz[j + width * i];
			imagen << " ";
		}
		imagen << "\n";
	}
	imagen.close();
}

void applyKernel(int image[],int result_image[], int kernel[][3], int size) {
	int prod;

	for (int i = 0; i < size * size; i++) {
		//for (int j = 0; j < 320; j++) {
		if( (i < size) || (i % size == 0) || ( (i + 1) % size == 0) || (i > size * (size - 1))) {
			result_image[i] = 0;
		}
		else {
			prod = kernel[0][0] * image[i - size - 1] +
				kernel[0][1] * image[i - size] +
				kernel[0][2] * image[i - size + 1] +
				kernel[1][0] * image[i - 1] +
				kernel[1][1] * image[i] +
				kernel[1][2] * image[i + 1] +
				kernel[2][0] * image[i + size - 1] +
				kernel[2][1] * image[i + size] +
				kernel[2][2] * image[i + size + 1];
			if (prod < 0)
				prod = 0;
			if (prod > 255)
				prod = 255;
			result_image[i] = prod;
			}
		//}	
	}
}
