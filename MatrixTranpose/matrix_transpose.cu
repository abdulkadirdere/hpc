#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

// int WIDTH = 512;
const int TILE_WIDTH = 32;

int randomNumberGeneration(int upperBound, int lowerBound) {
    int num = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
    return num;
}

int *createData(int *array, int num_element) {
    for (int i = 0; i < num_element; i++) {
        array[i] = randomNumberGeneration(9, 0);
    }
    return array;
}

int *createMatrix(int num_element) {
    int *array = (int *)malloc(num_element * sizeof(int *));
    // create synthetic data for matrix
    array = createData(array, num_element);
    return array;
}

int *allocateMatrix(int num_element) {
    int *array = (int *)malloc(num_element * sizeof(int *));
    return array;
}

void printArray(int *array, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", array[(i * width) + j]);
        }
        printf("\n");
    }
}

int *serialTranspose(int *array, int width){
    int *result = (int *)malloc(width*width * sizeof(int *));
    for (int i=0; i<width; i++){
        for (int j=0; j<width; j++){
            result[j*width +i] = array[i*width +j];
        }
    }
    return result;
}

__global__ void global_transpose_matrix(int *output_data, const int *input_data, int tile_width){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // blockDim = tile_width
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    for (int i = 0; i < blockDim.x; i+= width){
        output_data[x*width + (y+i)] = input_data[(y+i)*width + x];
    }
}

__global__ void shared_transpose_matrix(int *output_data, const int *input_data, int tile_width){

    __shared__ int tile[TILE_WIDTH] [TILE_WIDTH];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    for (int i = 0; i < blockDim.x; i+= width){
        tile[threadIdx.y+i][threadIdx.x] = input_data[(y+i)*width + x];
    }
    
    __syncthreads();

    // transpose the blocks
    x = blockIdx.y * blockDim.x + threadIdx.x; // swap block x to y
    y = blockIdx.x * blockDim.x + threadIdx.y; // swap block y to x

    for (int i=0; i< blockDim.x; i+=width){
        output_data[(y + i)*width + x] = tile[threadIdx.x][threadIdx.y+i];
    }
}

int main() {
    // matrix sizes 2^9, 2^10, 2^11, 2^12 = 512, 1024, 2048, 4096
    int matrix_sizes[] = {512, 1024, 2048, 4096};
    int WIDTH = matrix_sizes[0];
    int num_elemet = WIDTH * WIDTH;

    // create the matrix 
    int *h_input_matrix = createMatrix(num_elemet);
    int *h_result_matrix = allocateMatrix(num_elemet);
    int *d_input_matrix;
    int *d_output_matrix;

    // printArray(h_input_matrix, WIDTH);
    int memory_space_required = num_elemet * sizeof(int);

//-------------- Serial Matrix Transpose on CPU --------------//
    cudaEvent_t serial_start, serial_stop;
    cudaEventCreate(&serial_start);
    cudaEventCreate(&serial_stop);

    cudaEventRecord(serial_start);
    h_result_matrix = serialTranspose(h_input_matrix, WIDTH);
    cudaEventRecord(serial_stop);
    cudaEventSynchronize(serial_stop);

    float serial_time = 0;
    cudaEventElapsedTime(&serial_time, serial_start, serial_stop);
    printf("Serial Matrix Transpose Time: %3.6f ms \n", serial_time);
    // printArray(result_matrix, WIDTH);

//-------------- CUDA --------------//

    // allocate memory on device
    cudaMalloc((void **) &d_input_matrix, memory_space_required);
    cudaMalloc((void **) &d_output_matrix, memory_space_required);

    // CUDA timing of event
    cudaEvent_t global_start, global_stop, shared_start, shared_stop;
    cudaEventCreate(&global_start);
    cudaEventCreate(&global_stop);
    cudaEventCreate(&shared_start);
    cudaEventCreate(&shared_stop);

    // copy memory from host to device
    cudaMemcpy(d_input_matrix, h_input_matrix, memory_space_required, cudaMemcpyHostToDevice);

    // dimensions for the kernel
    dim3 Grid_Dim(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);
    dim3 Block_Dim(TILE_WIDTH, TILE_WIDTH);

//-------------- CUDA Matrix Transpose Global Memory --------------//
    // start the global memory kernel
    cudaEventRecord(global_start);
    global_transpose_matrix<<<Grid_Dim, Block_Dim>>>(d_output_matrix, d_input_matrix, TILE_WIDTH);
    cudaEventRecord(global_stop);
    cudaEventSynchronize(global_stop);

    float global_elapsedTime = 0;
    cudaEventElapsedTime(&global_elapsedTime, global_start, global_stop);
    printf("Global Memory Time elpased: %3.6f ms \n", global_elapsedTime );
    cudaEventDestroy(global_start);
    cudaEventDestroy(global_stop);

//-------------- CUDA Matrix Transpose Shared Memory --------------//
    // start the shared memory kernel
    cudaEventRecord(shared_start);
    shared_transpose_matrix<<<Grid_Dim, Block_Dim>>>(d_output_matrix, d_input_matrix, TILE_WIDTH);
    cudaEventRecord(shared_stop); 
    cudaEventSynchronize(shared_stop);
    
    float shared_elapsedTime = 0;
    cudaEventElapsedTime(&shared_elapsedTime, shared_start, shared_stop);

    printf( "Shared Memory Time elpased: %3.6f ms \n", shared_elapsedTime );
    cudaEventDestroy(shared_start);
    cudaEventDestroy(shared_stop);


    // copy back the results
    cudaMemcpy(h_result_matrix, d_output_matrix, memory_space_required, cudaMemcpyDeviceToHost);
    // printArray(h_result_matrix, WIDTH);

    cudaFree(h_input_matrix);
    cudaFree(h_result_matrix);
    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
}