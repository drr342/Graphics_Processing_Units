#include "cudaFunctions.h"

int maxThreadsPerSM(cudaDeviceProp * prop) {
    if (prop->major >= 3)
        return 2048;
    if (prop->major == 2)
        return 1536;
    if (prop->minor >= 2)
        return 1024;
    return 768;
}

int maxBlocksPerSM(cudaDeviceProp * prop) {
    if (prop->major >= 5)
        return 32;
    if (prop->major == 3)
        return 16;
    return 8;
}

__device__ int getBlockId(dim3 blockIdx, dim3 gridDim) {
    return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}

__device__ int getThreadId(int blockId, dim3 threadIdx, dim3 blockDim) {
    return blockId * (blockDim.x * blockDim.y * blockDim.z) +
        (threadIdx.z * (blockDim.x * blockDim.y)) +
        (threadIdx.y * blockDim.x) + threadIdx.x;
}
