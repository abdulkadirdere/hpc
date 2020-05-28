#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "helper/inc/helper_functions.h" // includes cuda.h and cuda_runtime_api.h
#include "helper/inc/helper_cuda.h" // helper functions for CUDA error check

const int size=512;
const int mask_size = 3;
const int offset = floor(mask_size/2);
const int padded_size = size + 2*offset;

// const double mask[3][3] = {
//     {1, 1, 1},
//     {1, 1, 1},
//     {1, 1, 1},
// };

// edge detection
const double mask[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

// sharpenning
// const double mask[3][3] = {
//     {-1, -1, -1},
//     {-1,  9, -1},
//     {-1, -1, -1},
// };

void printArray(double **array, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%3.6f ", array[i][j]);
        }
        printf("\n");
    }
}

double **allocateMatrix(int m, int n) {
    double **array = (double **)malloc(m * sizeof(double *));
    for (int i=0; i<m; i++){
        array[i] = (double *)malloc(n * sizeof(double)); 
    }

    int zero = 0; 
    for (int i = 0; i <  m; i++){
        for (int j = 0; j < n; j++) {
            array[i][j] = zero;
        }
    }

    return array;
}

double **convert2D(float *input, unsigned int width, unsigned int height) {
    double **array = (double **)malloc(width * sizeof(double *));
    for (int i=0; i<width; i++){
        array[i] = (double *)malloc(height * sizeof(double)); 
    }

    int value = 0; 
    for (int i = 0; i <  width; i++){
        for (int j = 0; j < height; j++) {
            array[i][j] = input[value];
            value++;
        }
    }
    return array;
}

float *convert1D(double **input, unsigned int width, unsigned int height) {
    unsigned int size = width * height * sizeof(float);
    float *array = (float *)malloc(size * sizeof(float));

    int value = 0; 
    for (int i = 0; i <  width; i++){
        for (int j = 0; j < height; j++) {
            array[value] = (float)input[i][j];
            value++;
        }
    }
    return array;
}

double **padArray(double **input, double **output) {
    int range = padded_size - offset;
    // printf("%d \n", range);

    // pad the array
    for (int i = offset; i < range; i++) {
        for (int j = offset; j < range; j++) {
            output[i][j] = input[i-offset][j-offset];
        }
    }
    return output;
}

double **unpad(double **input, double **output) {
    int range = padded_size - offset;

    // unpad the array
    for (int i = 0; i < range; i++) {
        for (int j = 0; j < range; j++) {
            output[i][j] = input[i+offset][j+offset];
        }
    }
    return output;
}

double applyMask(double **array, int row, int col){
    int n_size = offset * 2 + 1;

    // neighbours of given location
    double **neighbours = allocateMatrix(n_size, n_size);

    int range = padded_size - offset;

    neighbours[0][0] = array[row-1][col-1]; // top_left
    neighbours[0][1] = array[row-1][col]; // top_middle
    neighbours[0][2] = array[row-1][col+1]; //top_right

    neighbours[1][0] = array[row][col-1]; //middle_left
    neighbours[1][1] = array[row][col]; //middle_middle
    neighbours[1][2] = array[row][col+1]; //middle_right

    neighbours[2][0] = array[row+1][col-1]; //bottom_left
    neighbours[2][1] = array[row+1][col]; //bottom_middle
    neighbours[2][2] = array[row+1][col+1]; //bottom_right

    // printArray(neighbours, n_size, n_size);

    double **convolution = allocateMatrix(n_size, n_size);
    double value = 0;

    for (int r=0; r<3; r++){
        for(int c=0; c<3; c++){
            // printf("value: %3.6f \n", mask[1][1]);
            convolution[r][c] = mask[r][c] * neighbours[r][c];
            value = value + convolution[r][c];
        }
    }
    // printf("value: %3.6f \n", value);
    // printArray(convolution, offset, offset);

    return value;
}

double **serial_convolution(double **input, double **output){
    int range = padded_size - offset;
    // printf("range: %d \n", range);

    for (int i = offset; i<range; i++){
        for (int j = offset; j<range; j++){
            output[i][j] = applyMask(input, i, j);
        }
    }
    return output;
}

int main(int argc, char **argv){
    int devID = findCudaDevice(0, 0);
    cudaGetDeviceProperties(0, 0);

    const char *imageFilename = "lena_bw.pgm";

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // convert image to 2D
    double **image =  convert2D(hData, width, height);
    printf("Input image \n");
    printArray(image, 10, 10);

    // allocate space for padded image
    double **padded = allocateMatrix(padded_size, padded_size);
    padded = padArray(image, padded);
    // printf("Padded image \n");
    // printArray(padded, 10, 10);

    // convolution results
    double **output = allocateMatrix(padded_size, padded_size);
    output = serial_convolution(padded, output);
    // printf("Convolution image \n");
    // printArray(output, 10, 10);

    // unpad the array
    double **unpadded = allocateMatrix(padded_size, padded_size);
    unpadded = unpad(output, unpadded);
    printf("unpadded image \n");
    printArray(unpadded, 10, 10);

    // update array
    float *result_image;
    result_image = convert1D(unpadded, width, height);

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, result_image, width, height);
    printf("Wrote '%s'\n", outputFilename);

    free(image);
    free(padded);
    free(output);
    free(unpadded);
    free(result_image);

}