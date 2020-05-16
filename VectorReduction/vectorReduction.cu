#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

const int THREAD_SIZE = 1024 * sizeof(int);

int randomNumberGeneration(int upperBound, int lowerBound) {
    // creates a random integer within the bounds
    int num = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
    return num;
}

int *createData(int *vector, int num_element) {
    // creates random integer data for the vector
    for (int i = 0; i < num_element; i++) {
        vector[i] = randomNumberGeneration(9, 0);
    }
    return vector;
}

int *createVector(int num_element){
    int *vector = (int *)malloc(num_element * sizeof(int *));
    // create synthetic data for vector
    vector = createData(vector, num_element);
    return vector;
}

int *allocateVector(int num_element) {
    // allocates space for the vector
    int *vector = (int *)malloc(num_element * sizeof(int *));
    return vector;
}

int serialVectorSum(int *h_input_vector, int num_element){
    int sum = 0;
    // sums each element of the vector until end of number of elements
    for (int i=0; i < num_element; i++){
        sum = sum + h_input_vector[i];
    }
    return sum;
}

void printVector(int *vector, int num_element) {
    for (int i = 0; i < num_element; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

__global__ void globalVectorSum(int *d_output, int *d_input){
    // get the total thread number
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // get the thread number in each block
    int tdx = threadIdx.x;
    
    // divide block into 2 sections to work on
    // keep dividing the block into 2 until only one element is remaing
    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (tdx < s){
            d_input[i] += d_input[i + s];
        }
        __syncthreads();
    }

    // thread 0 will write the results from block dividing to output
    if (tdx == 0){
        d_output[blockIdx.x] = d_input[i];
    }
}

__global__ void sharedVectorSum(int *d_output, int *d_input){
    // shared data is allocated from kernel
    __shared__ int shared_data[THREAD_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tdx = threadIdx.x;

    // copy all the values of input data from global memory to shared memory
    shared_data[tdx] = d_input[i];
    __syncthreads(); // boundary to wait for all threads to finish copying

    // do reduction in shared memory. Similar to global memory method
    // keep dividing the block into 2 blocks until only one element is left
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if (tdx < stride){
            shared_data[tdx] += shared_data[tdx + stride];
        }
        __syncthreads(); // boundary to wait for all threads to finish dividing
    }

    // thread 0 will write the results from shared memory to global memory
    if (tdx == 0){
        d_output[blockIdx.x] = shared_data[0];
    }
}

int sum(int *h_output_vector, int num_element){
    int sum = 0;
    for (int i = 0; i<num_element; i++){
        sum += h_output_vector[i];
    }
    return sum;
}

int main(){
    const int num_element = 1024000;

    // Host memory allocation
    int *h_input_vector = createVector(num_element);
    int *h_output_vector = allocateVector(num_element);

    // printVector(h_input_vector, num_element);
    int memory_space_required = num_element * sizeof(int);

    //-------------- Serial Vector Summation CPU --------------//
    cudaEvent_t serial_start, serial_stop;
    cudaEventCreate(&serial_start);
    cudaEventCreate(&serial_stop);

    cudaEventRecord(serial_start);
    int serial_sum = serialVectorSum(h_input_vector, num_element);
    cudaEventRecord(serial_stop);
    cudaEventSynchronize(serial_stop);

    float serial_time = 0;
    cudaEventElapsedTime(&serial_time, serial_start, serial_stop);
    // printf("Serial Sum: %d\n", serial_sum);
    cudaEventDestroy(serial_start);
    cudaEventDestroy(serial_stop);

//-------------- CUDA Vector Summation Global Memory --------------//
    // Device memory allocation
    int *d_input_vector;
    int *d_output_vector;

    cudaMalloc((void **) &d_input_vector, memory_space_required);
    cudaMalloc((void **) &d_output_vector, memory_space_required);

    // CUDA timing of event
    cudaEvent_t global_start, global_stop, shared_start, shared_stop;
    cudaEventCreate(&global_start);
    cudaEventCreate(&global_stop);
    cudaEventCreate(&shared_start);
    cudaEventCreate(&shared_stop);

   // dimensions for the kernel
   int MAX_THREADS = 1024;
   int NUM_THREADS = MAX_THREADS;
   int NUM_BLOCKS = num_element / MAX_THREADS;
   if (NUM_BLOCKS == 0 ){
       NUM_BLOCKS = 1;
   }

    //-------------- CUDA Vector Summation Global Memory --------------//
    // copy memory from host to device
    cudaMemcpy(d_input_vector, h_input_vector, memory_space_required, cudaMemcpyHostToDevice);
    // global vector kernel
    cudaEventRecord(global_start);
    globalVectorSum<<<NUM_BLOCKS, NUM_THREADS>>>(d_output_vector, d_input_vector);
    cudaEventRecord(global_stop);
    cudaEventSynchronize(global_stop);

    float global_elapsedTime = 0;
    cudaEventElapsedTime(&global_elapsedTime, global_start, global_stop);
    cudaEventDestroy(global_start);
    cudaEventDestroy(global_stop);
    cudaMemcpy(h_output_vector, d_output_vector, memory_space_required, cudaMemcpyDeviceToHost);
    
    int global_sum = sum(h_output_vector, num_element);
    // printf("Global Memory Sum: %d \n", global_sum);

    //-------------- Free Memory --------------//
    free(h_input_vector);
    free(h_output_vector);
    cudaFree(d_input_vector);
    cudaFree(d_output_vector);
}