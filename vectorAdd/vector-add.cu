#include <stdio.h>
#define CUDA_CALL( X ) { \
    cudaError_t mycudaerror = ( X ) ; \
    if (mycudaerror != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s [%s:%d]\n", cudaGetErrorString(mycudaerror), __FILE__, __LINE__) ;\
    }}


__global__ void vectorAdd(double *a, double *b, double *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vectorAdd_kernel(double *a, double *b, double *c, int n)
{
    dim3 block(32);
    dim3 grid((n + 32 - 1) / 32);

    vectorAdd<<<grid,block>>>(a, b, c, n);
    CUDA_CALL(cudaGetLastError());
}
