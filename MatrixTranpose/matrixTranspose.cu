#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <iostream>


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

int *sequential_transpose(int *array, int width){
    // allocate space for the results array
    int *result = (int *)malloc(width*width * sizeof(int *));
    // swap i and j values (x and y) in the matrix to produce the transpose
    // result maps to result(j,i) = input(i,j)
    for (int i=0; i<width; i++){
        for (int j=0; j<width; j++){
            result[j*width +i] = array[i*width +j];
        }
    }
    return result;
}

__global__ void global_transpose_matrix(int *output_data, const int *input_data){
    // x =  get threads in x direction i.e. columns
    // y = get threads in y direction i.e. rows
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x; 

    // swap x and x values in the matrix to produce the transpose
    // results map to output(y,x) = input(x,y)
    for (int i = 0; i < blockDim.x; i+= width){
        output_data[x*width + (y+i)] = input_data[(y+i)*width + x];
    }
}

__global__ void shared_transpose_matrix(int *output_data, const int *input_data){
    // creating tiles in the shared memory for faster computation
    // each tile is created of the preset size i.e. 32 in this case
    __shared__ int tile[TILE_WIDTH] [TILE_WIDTH+1];
    // x =  get threads in x direction i.e. columns
    // y = get threads in y direction i.e. rows
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    // swap x and x values in the tile to produce the transposed tiled
    // tile maps to (y,x) = input(x,y)
    // we save these tiles in transposed hence it is (y,x)
    // this will help us to elimanate the usage of thread synching for output 
    for (int i = 0; i < blockDim.x; i+= width){
        tile[threadIdx.y+i][threadIdx.x] = input_data[(y+i)*width + x];
    }
    // thread barrier. All threads must reach this barrier before continuing
    __syncthreads();

    // transpose the blocks for writing to output matrix
    x = blockIdx.y * blockDim.x + threadIdx.x; // swap block x to y
    y = blockIdx.x * blockDim.x + threadIdx.y; // swap block y to x

    // write to output data from tiles (shared memory) to the global memory
    for (int i=0; i< blockDim.x; i+=width){
        output_data[(y + i)*width + x] = tile[threadIdx.x][threadIdx.y+i];
    }
}

int main() {
    // matrix sizes 2^9, 2^10, 2^11, 2^12 = 512, 1024, 2048, 4096
    int matrix_sizes[] = {256, 512, 1024, 2048, 4096};
        int WIDTH = matrix_sizes[4];
        int num_element = WIDTH * WIDTH;

        // create the matrix and fill it with random numbers
        int *h_input_matrix = createMatrix(num_element);
        int *h_result_matrix = allocateMatrix(num_element);
        int *d_input_matrix;
        int *d_output_matrix;

        // printArray(h_input_matrix, WIDTH);
        int memory_space_required = num_element * sizeof(int);

    //-------------- Serial Matrix Transpose on CPU --------------//
        cudaEvent_t serial_start, serial_stop;
        cudaEventCreate(&serial_start);
        cudaEventCreate(&serial_stop);

        cudaEventRecord(serial_start);
        h_result_matrix = sequential_transpose(h_input_matrix, WIDTH);
        cudaEventRecord(serial_stop);
        cudaEventSynchronize(serial_stop);

        float serial_time = 0;
        cudaEventElapsedTime(&serial_time, serial_start, serial_stop);
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
        global_transpose_matrix<<<Grid_Dim, Block_Dim>>>(d_output_matrix, d_input_matrix);
        cudaEventRecord(global_stop);
        cudaEventSynchronize(global_stop);

        float global_elapsedTime = 0;
        cudaEventElapsedTime(&global_elapsedTime, global_start, global_stop);
        cudaEventDestroy(global_start);
        cudaEventDestroy(global_stop);

    //-------------- CUDA Matrix Transpose Shared Memory --------------//
        // start the shared memory kernel
        cudaEventRecord(shared_start);
        shared_transpose_matrix<<<Grid_Dim, Block_Dim>>>(d_output_matrix, d_input_matrix);
        cudaEventRecord(shared_stop); 
        cudaEventSynchronize(shared_stop);
        
        float shared_elapsedTime = 0;
        cudaEventElapsedTime(&shared_elapsedTime, shared_start, shared_stop);

        cudaEventDestroy(shared_start);
        cudaEventDestroy(shared_stop);

        // copy back the results
        cudaMemcpy(h_result_matrix, d_output_matrix, memory_space_required, cudaMemcpyDeviceToHost);
        // printArray(h_result_matrix, WIDTH);

         //-------------- CUDA Performance Metrics --------------//
         float num_ops= num_element; // every element swap once
  
         float serial_throughput = num_ops / (serial_time / 1000.0f) / 1000000000.0f;
         float global_throughput = num_ops / (global_elapsedTime / 1000.0f) / 1000000000.0f;
         float shared_throughput = num_ops / (shared_elapsedTime / 1000.0f) / 1000000000.0f;
 
         std::cout << "Matrix size: " << WIDTH << "x" << WIDTH << std::endl;
         std::cout << "Tile size: " << TILE_WIDTH << "x" << TILE_WIDTH << std::endl;
 
         printf("Serial Matrix Transpose Time: %3.6f ms \n", serial_time);
         printf("Global Memory Time elpased: %3.6f ms \n", global_elapsedTime);
         printf( "Shared Memory Time elpased: %3.6f ms \n", shared_elapsedTime );
 
         std::cout << "\nSpeedup of global memory kernel (CPU/GPU): " << serial_time / global_elapsedTime << " ms" << std::endl;
         std::cout << "Speedup of shared memory kernel (CPU/GPU): " << serial_time / shared_elapsedTime << " ms" << std::endl;
       
         std::cout << "\nThroughput of serial implementation: " << serial_throughput << " GFLOPS" << std::endl;
         std::cout << "Throughput of global memory kernel: " << global_throughput << " GFLOPS" << std::endl;
         std::cout << "Throughput of shared memory kernel: " << shared_throughput << " GFLOPS" << std::endl;
         std::cout << "Performance improvement: simple over global " << serial_throughput / global_throughput << "x" << std::endl;
         std::cout << "Performance improvement: simple over shared " << serial_throughput / shared_throughput << "x" << std::endl;
         std::cout << "Performance improvement: global over shared " << global_throughput / shared_throughput << "x" << std::endl;

        //-------------- Free Memory --------------//
        free(h_input_matrix);
        free(h_result_matrix);
        cudaFree(d_input_matrix);
        cudaFree(d_output_matrix);
        printf("\n");
}