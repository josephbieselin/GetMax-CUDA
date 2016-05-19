#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                  RUN THIS EVERY TIME YOU LOG INTO CIMS:                    //
//                                                                            //
//                  module load mpi/mpich-x86_64                              //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



int main(int argc, char *argv[])
{
  int nDevices;

  cudaGetDeviceCount(&nDevices);

  for (int i = 0; i < nDevices; ++i)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %u\n", i);
    printf("\tDevice Name: %s\n", prop.name);
    printf("\tCompute Capability: %u.%u\n", prop.major, prop.minor);
    printf("\tMemory Clock Rate (KHz): %u\n", prop.memoryClockRate);
    printf("\tMemory Bus Width (bits): %u\n", prop.memoryBusWidth);
    // printf("\tPeak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("\t%s: %u\n", "Total Global Memory", prop.totalGlobalMem);
    printf("\t%s: %u\n", "Shared Memory Per Block", prop.sharedMemPerBlock);
    printf("\t%s: %u\n", "Registers Per Block", prop.regsPerBlock);
    printf("\t%s: %u\n", "Warp Size", prop.warpSize);
    printf("\t%s: %u\n", "Max Threads Per Block", prop.maxThreadsPerBlock);
    printf("\t%s: %u\n", "Max Threads Per MultiProcessor", prop.maxThreadsPerMultiProcessor);
    printf("\t%s: %u, %u, %u\n", "Max Threads Dimensions", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("\t%s: %u, %u, %u\n", "Max Grid Size", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\t%s: %u\n", "Total Constant Memory", prop.totalConstMem);
    printf("\t%s: %u\n", "MultiProcessor Count", prop.multiProcessorCount);
    printf("\t%s: %u\n", "L2 Cache Size", prop.l2CacheSize);
    // printf("\t%s: %u\n", "Concurrent Kernels", prop.concurrentKernels);
  }

   exit(0);
}
