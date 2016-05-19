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

/* Linx command 'time' was used to report the REAL part (neither usr nor sys)
 * Used as such:
 * $ time ./progname [arguments]
 */


/* Constants from cudaGetDeviceProperties() */
/* In order to not slow down the program, these values were found for the
 * device being used beforehand.
 * Otherwise, these would be set by calling cudaGetDeviceProperties()
 * and setting them accordingly.
 */

// Max Grid Size
long MAX_GRIDX = 2147483647;
long MAX_GRIDY = 65535;
long MAX_GRIDZ = 65535;

// Max Thread Dimensions
long MAX_THREADX = 1024;
long MAX_THREADY = 1024;
long MAX_THREADZ = 64;

// Max Threads per Multiprocessor
long MAX_THREADS_PER_MULTIPROCESSOR = 2048;

// Max Threads per Block
long MAX_THREADS_PER_BLOCK = 1024;



#define ELEMENT_N 256

/* Functions in use */
__global__ void getlocalmaxcu(long Nd[], long partialMaxes[], long size); // GPU kernel to get a portion of possible max elements
__global__ void getmaxcu(long partialMaxes[], long *max, long size); // GPU kernel that gets THE max element
long getmax(long num[], long size); // sequential getmax function called by host
long myCeil(long x, long y);



int main(int argc, char *argv[])
{
  // WAS RUNNING INTO A BLOCKED EXECUTION ON DEVICE 0 WHEN GPU CODE WAS TRYING TO START
  cudaSetDevice(1);

  long size = 0;  // The size of the array
  long i;  // loop index
  long * numbers; //pointer to the array
  long * hostMax = (long*) malloc(sizeof(long)); // pointer to place the max element
   
  if(argc !=2)
  {
    printf("usage: maxseq num\n");
    printf("num = size of the array\n");
    exit(1);
  }
   
  size = atol(argv[1]);

  numbers = (long *)malloc(size * sizeof(long));
  if( !numbers )
  {
    printf("Unable to allocate mem for an array of size %ld\n", size);
    exit(1);
  }    

  srand(time(NULL)); // setting a seed for the random number generator
  // Fill-up the array with random numbers from 0 to size-1 
  for( i = 0; i < size; i++)
    numbers[i] = rand() % size;



  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  //                                                                            //
  //                                GPU STUFF                                   //
  //                                                                            //
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  // Allocate memory on the device for the numbers
  long * Nd; // pointer to the array for the device
  long * max; // pointer to the max element
  long * partialMaxes; // pointer to potential maxes
  long * tempPtr; // temporary long pointer to hold memory location of another array
  long memSize = size * sizeof(long);

  cudaMalloc((void**) &Nd, memSize);
  cudaMalloc((void**) &max, sizeof(long));

  // Copy over the contents to the device memory
  cudaMemcpy(Nd, numbers, memSize, cudaMemcpyHostToDevice);


  /* NOTE: cudaMemcpy is an Asynchronous transfer so host is not blocked */


  long gridX = 1;

  // X = 256 because it is a multiple of the current warp size and allows for 8 Blocks to be put into an SM
  long blockX = ELEMENT_N;

  // calculate the number of blocks we'll need
  gridX = myCeil(size, blockX);

  // each block will place its local max into partialMaxes
  long partialMemSize = gridX * sizeof(long);
  cudaMalloc((void**) &partialMaxes, partialMemSize);




  // dimensions to be passed to the kernel
  dim3 dimGrid(gridX);
  dim3 dimBlock(blockX);

  // kernel calls are Asynchronous so host is not blocked (it continues execution immediately)
  getlocalmaxcu<<<dimGrid, dimBlock>>>(Nd, partialMaxes, size);

  /* NOTE: kernel calls are Asynchronous so host is not blocked */


  /* partialMaxes now has size gridX where each element is the possible max at this moment */


  // while the array with local maxes is larger than the block size, compute
  while (gridX > blockX)
  {
    // size of the local maxes array is gridX
    size = gridX;

    // calculate the new number of blocks we'll need
    gridX = myCeil(size, blockX);

    /* Switch the memory pointers between Nd and partialMaxes:
     * partialMaxes now contains the elements we need to check
     * for being the max (originally it was Nd that had all elements).
     * After the kernel executes, the new partialMaxes memory location
     * which is Nd when swapped, will contain a smaller subset of
     * possible max elements.
     * These switches keep happening until there are less than
     * blockDim.x elements to check.
     */
    tempPtr = Nd;
    Nd = partialMaxes;
    partialMaxes = tempPtr;

    // new dimensions to be passed to the kernel
    // dim3 dimGrid(gridX);
    dimGrid.x = gridX;

    // kernel calls are Asynchronous so host is not blocked (it continues execution immediately)
    getlocalmaxcu<<<dimGrid, dimBlock>>>(Nd, partialMaxes, size);

    /* NOTE: kernel calls are Asynchronous so host is not blocked */
  }

  // partialMaxes now contains local maxes for a number of elements less than ELEMENT_N aka blockX
  // launch the final kernel to get the overall max
  dimGrid.x = 1;
  dimBlock.x = gridX;

  // get the TRUE max
  getmaxcu<<<dimGrid, dimBlock>>>(partialMaxes, max, gridX);
  
  /* NOTE: kernel calls are Asynchronous so host is not blocked */

  // Copy over the device contents back to host memory
  cudaMemcpy(hostMax, max, sizeof(long), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize(); // this will force host to wait until device completes memory transfer


  // Print out the GPU's result
  printf(" The maximum number in the array is: %ld\n", *hostMax);


  // Free memory on the host
  free(numbers);
  free(hostMax);
  
  // Free memory on the device
  cudaFree(Nd);
  cudaFree(max);
  cudaFree(partialMaxes);

  exit(0);
}


/* Computes the local maxes for each array.
 * The array Nd contains a long list of random numbers.
 * Each block's local max is put into its block ID's index in partialMaxes.
 */
__global__ void getlocalmaxcu(long Nd[], long partialMaxes[], long size)
{
  long tx = threadIdx.x;

  // unique Thread ID in the grid
  long id = ( blockIdx.x * blockDim.x ) + tx;

  // shared (per block) array to store the local maxes of each thread
  __shared__ long tempMaxes[ELEMENT_N]; // shared memory size set on kernel call in host as the block dimension

  long localMax, compVal;


  if (id < size)
  {
    localMax = Nd[id];
    tempMaxes[tx] = localMax;
  }
  else
  {
    tempMaxes[tx] = 0;
  }

  // threads will compare their localMax with the max element
  // of the other threads in the block stored in tempMaxes.
  // stride is used to determine which threads should be doing
  // comparisons and updating their localMax/tempMaxes values.
  for (long stride = ELEMENT_N / 2; stride > 0; stride >>= 1)
  {
    // wait until all threads have updated values
    __syncthreads();

    if (tx < stride)
    {
      if ( (id + stride) < size )
      {
        compVal = tempMaxes[tx + stride];
        if (compVal > localMax)
        {
          localMax = compVal;
          tempMaxes[tx] = localMax;
        }
      }
    }
  }

  // Only 1 thread should push an update to the global array
  if (tx == 0)
  {
    // Place the max element from this block (in Thread 0) into the partialMaxes array
    partialMaxes[blockIdx.x] = localMax;
  }
}


__global__ void getmaxcu(long partialMaxes[], long *max, long size)
{
  // Only need threadIdx.x since this function only gets called with 1 block
  long tx = threadIdx.x;


  // shared (per block) array to store the local maxes of each thread
  __shared__ long tempMaxes[ELEMENT_N]; // shared memory size set on kernel call in host as the block dimension

  long localMax, compVal;


  if (tx < size)
  {
    localMax = partialMaxes[tx];
    tempMaxes[tx] = localMax;
  }
  else
  {
    tempMaxes[tx] = 0;
  }
  
  // threads will compare their localMax with the max element
  // of the other threads in the block stored in tempMaxes.
  // stride is used to determine which threads should be doing
  // comparisons and updating their localMax/tempMaxes values.
  for (long stride = ELEMENT_N / 2; stride > 0; stride >>= 1)
  {
    // wait until all threads have updated values
    __syncthreads();

    if (tx < stride)
    {
      if ( (tx + stride) < size )
      {
        compVal = tempMaxes[tx + stride];
        if (compVal > localMax)
        {
          localMax = compVal;
          tempMaxes[tx] = localMax;
        }
      }
    }
  }

  // Only 1 thread should push an update to the global max
  if (tx == 0)
    *max = localMax;
}


/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
long getmax(long num[], long size)
{
  long i;
  long max = num[0];

  for(i = 1; i < size; i++)
  if(num[i] > max)
     max = num[i];

  return( max );

}


/* Returns ceil(x / y) */
long myCeil(long x, long y)
{
  long z = x / y;

  if ((x % y) != 0)
    z++;

  return z;
}