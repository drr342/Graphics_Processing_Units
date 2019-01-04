#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char * argv[]) {
    int N = atoi(argv[1]);
    int * arr = (int*) calloc(N, sizeof(int));

    size_t size = N * sizeof(int);
    int * dArr;
    cudaMalloc(&dArr, size);
    cudaMemset(dArr, 0, size);
    cudaMemcpy(arr, dArr, size, cudaMemcpyDeviceToHost);
    free(arr);
    cudaFree(dArr);

    return 0;
}
