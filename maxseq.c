#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long getmax(long *, long);

int main(int argc, char *argv[])
{
   long size = 0;  // The size of the array
   long i;  // loop index
   long * numbers; //pointer to the array
    
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
   
    printf(" The maximum number in the array is: %ld\n", 
           getmax(numbers, size));

    free(numbers);
    exit(0);
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
