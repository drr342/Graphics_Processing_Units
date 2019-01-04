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
 *		cuda5: nvcc -o heatdist -arch=sm_35 heatdist.cout
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cudaFunctions.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);
float checksum(float * playground, unsigned int N);
int setDimensions(int N, dim3 * grid, dim3 * block, cudaDeviceProp * prop);
__global__ void calculate(float * dPlayground, float * dTemp, int * dParams);

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
    int * dParams;
    size_t sizeN = sizeof(float) * N * N;
    size_t sizeParams = sizeof(int) * 2;

    cudaMalloc(&dPlayground, sizeN);
    cudaMalloc(&dTemp, sizeN);
    cudaMalloc(&dParams, sizeParams);

    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    dim3 dimGrid, dimBlock;
    int split = setDimensions(N - 2, &dimGrid, &dimBlock, &prop);
    int params[2] = {N, split};

    cudaMemcpy(dPlayground, playground, sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(dTemp, playground, sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(dParams, params, sizeParams, cudaMemcpyHostToDevice);

    for (int k = 0; k < iterations; k++) {
        calculate<<<dimGrid, dimBlock>>>(dPlayground, dTemp, dParams);
        cudaMemcpy(dPlayground, dTemp, sizeN, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(playground, dPlayground, sizeN, cudaMemcpyDeviceToHost);

	cudaFree(dPlayground);
	cudaFree(dTemp);
    cudaFree(dParams);
}

int setDimensions(int N, dim3 * grid, dim3 * block, cudaDeviceProp * prop) {
    int maxTpSM = maxThreadsPerSM(prop);
    int maxBpSM = maxBlocksPerSM(prop);
    int TpB = prop->maxThreadsPerBlock / maxBpSM;
    // while (maxTpSM % TpB != 0) TpB /= 2;
    if (TpB > N) TpB = N;
    int maxBpGy = prop->maxGridSize[1];
    int BpGy = N;
    int BpGx = (int) ceil((float)N / TpB);
    while (BpGy > maxBpGy) {
        BpGy = (int) ceil(BpGy / 2.0);
        BpGx *= 2;
    }
    grid->x = BpGx;
    grid->y = BpGy;
    block->x = TpB;
    return TpB * (int) ceil((N - 2.0) / TpB);
}

__global__ void calculate(float * dPlayground, float * dTemp, int * dParams) {
    int dN = dParams[0];
    int dSplit = dParams[1];
    int blockId = getBlockId(blockIdx, gridDim);
    int threadId = getThreadId(blockId, threadIdx, blockDim);
    int row = threadId / dSplit + 1;
    int col = threadId % dSplit + 1;
    if (col > dN - 2 || row > dN - 2) return;

    dTemp[index(row, col, dN)] =
        (dPlayground[index(row - 1, col, dN)] +
        dPlayground[index(row + 1, col, dN)] +
        dPlayground[index(row, col - 1, dN)] +
        dPlayground[index(row, col + 1, dN)]) / 4.0;
}
