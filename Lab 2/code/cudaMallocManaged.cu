#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char * argv[]) {
    int N = atoi(argv[1]);

    size_t size = N * sizeof(int);
    int * dArr;
    cudaMallocManaged(&dArr, size);
    cudaMemset(dArr, 0, size);
    cudaDeviceSynchronize();
    cudaFree(dArr);

    return 0;
}
