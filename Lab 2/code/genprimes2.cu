/*
 *  PRIME NUMBERS GENERATOR
 *
 *  Last name: RIVERA RUIZ
 *  First name: DANIEL
 *  Net ID: drr342
 *
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))

__global__ void calculate (int *, int);
void toFile(int *, int);

int main(int argc, char * argv[]) {
    int N = atoi(argv[1]);
    size_t size = sizeof(int) * (N - 1);

    int * primes = (int*) malloc(size);
    int * dPrimes;
    cudaMalloc(&dPrimes, size);
    cudaMemset(dPrimes, 0, size);

    dim3 dimGrid(ceil(N / 1000.0));
    dim3 dimBlock(min(N, 1000));
    calculate<<<dimGrid, dimBlock>>>(dPrimes, N);
    cudaMemcpy(primes, dPrimes, size, cudaMemcpyDeviceToHost);

    toFile(primes, N);

    free(primes);
    cudaFree(dPrimes);
    return 0;
}

__global__ void calculate(int * dPrimes, int N) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int mult = 2 * threadId;
    while (mult <= N) {
        dPrimes[mult - 2] = 1;
        mult += threadId;
    }
}

void toFile(int * primes, int N) {
    char * fileName = (char*) malloc(13 * sizeof(char));
    sprintf(fileName, "%d.txt", N);
    FILE * fp;
    fp = fopen(fileName,"w");
    for (int i = 0; i < N - 1; i++) {
        if (primes[i] == 0)
            fprintf(fp, "%d ", i + 2);
    }
    fprintf(fp, "\n");
    fclose(fp);
    free(fileName);
}
