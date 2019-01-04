/*
 *  Please write your name and net ID below
 *
 *  Last name: RIVERA RUIZ
 *  First name: DANIEL
 *  Net ID: drr342
 *
 */


/*
 * This file contains the code for doing the heat distribution problem.
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s) that you need to write too.
 *
 * You compile with:
 * 		cuda2: nvcc -o heatdist -arch=sm_52 heatdist.cu
 *		cuda5: nvcc -o heatdist -arch=sm_35 heatdist.cu
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);
float checksum(float * playground, unsigned int N);
__global__ void calculate(float * dPlayground, float * dTemp, int * dN);

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;

  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground;

  // to measure time taken by a specific part of the code
  double time_taken;
  clock_t start, end;

  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU execution\n");
    exit(1);
  }

  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);


  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }

  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements to 70F
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 70;

  for(i = 0; i < N; i++)
    playground[index(i,0,N)] = 70;

  for(i = 0; i < N; i++)
    playground[index(i,N-1, N)] = 70;

  for(i = 0; i < N; i++)
    playground[index(N-1,i,N)] = 70;

  // from (0,10) to (0,30) inclusive are 100F
  for(i = 10; i <= 30; i++)
    playground[index(0,i,N)] = 100;

   // from (n-1,10) to (n-1,30) inclusive are 150F
  for(i = 10; i <= 30; i++)
    playground[index(N-1,i,N)] = 150;

  if( !type_of_device ) // The CPU sequential version
  {
    start = clock();
    seq_heat_dist(playground, N, iterations);
    end = clock();
  }
  else  // The GPU version
  {
     start = clock();
     gpu_heat_dist(playground, N, iterations);
     end = clock();
  }

  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

  printf("Checksum: %f\n", checksum(playground, N));
  printf("Time taken for %s is %lf\n", type_of_device == 0? "CPU" : "GPU", time_taken);

  free(playground);

  return 0;
}

/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;

  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;

  float * temp;
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }

  num_bytes = N*N*sizeof(float);

  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);

  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] +
	                      playground[index(i+1,j,N)] +
			      playground[index(i,j-1,N)] +
			      playground[index(i,j+1,N)])/4.0;

    /* Move new values into old values */
    memcpy((void *)playground, (void *) temp, num_bytes);
  }

}

float checksum(float * playground, unsigned int N) {
    float sum = 0;
    int i, j;
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            sum += playground[index(i, j, N)];
    return sum;
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
    float * dPlayground, * dTemp;
    int * dN;
    size_t sizeN = sizeof(float) * N * N;
	
    cudaMalloc(&dPlayground, sizeN);
    cudaMalloc(&dTemp, sizeN);
    cudaMalloc(&dN, sizeof(int));

    cudaMemcpy(dPlayground, playground, sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(dTemp, playground, sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(dN, &N, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(N - 2, (int) ceil((N - 2.0) / 32.0));
    dim3 dimBlock(32);
    for (int k = 0; k < iterations; k++) {
        calculate<<<dimGrid, dimBlock>>>(dPlayground, dTemp, dN);
        cudaMemcpy(dPlayground, dTemp, sizeN, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(playground, dPlayground, sizeN, cudaMemcpyDeviceToHost);

	cudaFree(dPlayground);
	cudaFree(dTemp);
	cudaFree(dN);
}

__global__ void calculate(float * dPlayground, float * dTemp, int * dN) {
    int row = blockIdx.x + 1;
    int col = blockIdx.y * 32 + threadIdx.x + 1;
    if (col > *dN - 2) return;

    dTemp[index(row, col, *dN)] =
        (dPlayground[index(row - 1, col, *dN)] +
        dPlayground[index(row + 1, col, *dN)] +
        dPlayground[index(row, col - 1, *dN)] +
        dPlayground[index(row, col + 1, *dN)]) / 4.0;
}
