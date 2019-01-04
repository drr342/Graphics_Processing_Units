#ifndef MY_CUDA_LIB_H
#define MY_CUDA_LIB_H

    #include <cuda.h>

    #define DEVICE 0
    #define WARP_SIZE 32

    typedef struct cudaDeviceProp cudaDeviceProp;
    int maxThreadsPerSM(cudaDeviceProp*);
    int maxBlocksPerSM(cudaDeviceProp*);
    __device__ int getBlockId(dim3, dim3);
    __device__ int getThreadId(int, dim3, dim3);

#endif
