// This example demonstrates how to 
// query about the properties of a device

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


int main(void)
{
  int dev_count, driverVersion = 0, runtimeVersion = 0;;
  cudaGetDeviceCount(&dev_count);
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10,
             runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  cudaDeviceProp dev_prop;
  for(int i=0; i<dev_count;i++){
		cudaGetDeviceProperties(&dev_prop,i);
		printf("---Device %d---\n", i);
		printf("Name: \"%s\"\n", dev_prop.name);
		printf("CUDA Capability Major/Minor version number: %d.%d\n", dev_prop.major, dev_prop.minor);
		printf("--- Memory information for device ---\n");
		printf("Total global mem: %.0f MB\n", dev_prop.totalGlobalMem/1048576.0f);
		printf("Total constant mem: %lu B\n", dev_prop.totalConstMem);
		printf("The size of shared memory per block: %lu B\n", dev_prop.sharedMemPerBlock);
		printf("The maximum number of registers per block: %d\n", dev_prop.regsPerBlock);
		printf("The number of SMs on the device: %d\n", dev_prop.multiProcessorCount);
		printf("The number of threads in a warp: %d\n", dev_prop.warpSize);
		printf("The maximal number of threads allowed in a block: %d\n", dev_prop.maxThreadsPerBlock);
		printf("Max thread dimensions (x,y,z): (%d, %d, %d)\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
		printf("Max grid dimensions (x,y,z): (%d, %d, %d)\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
	}	
}

