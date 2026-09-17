#include <stdio.h>
#include <cassert>

void initMatrix(double *a, int n, int m, double c)
{
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < m; ++j)
        {
            a[i * n + j] = c;
        }
    }
}

#define CUDA_CHECK( X ) { \
   cudaError_t mycudaerror = ( X ); \
   if (mycudaerror != cudaSuccess) {\
        fprintf(stderr, "CUDA Error: %s [%s:%d]\n" \
        , cudaGetErrorString(mycudaerror), __FILE__, __LINE__); \
    }}

// c[i] = a[i] + q * b[i];
__global__ void triadBandwidth(double *a, double *b, double *c, double q, int n, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m)
    {
        c[i * m + j] = a[i * m + j] + q * b[i * m + j];
    }
}
int main(int argc, char **argv)
{

    const int N = 7000, M = 7000;
    int tile_width = 32;
    dim3 block(tile_width, tile_width);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    const double q = 60.0;
    double *h_a, *h_b, *h_c;
    h_a = (double *)malloc(N * M * sizeof(double));
    h_b = (double *)malloc(N * M * sizeof(double));
    h_c = (double *)malloc(N * M * sizeof(double));

    double *dev_a, *dev_b, *dev_c;

    CUDA_CHECK(cudaMalloc(&dev_a, N * M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_b, N * M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_c, N * M * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dev_a, h_a, N * M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, h_b, N * M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_c, h_c, N * M * sizeof(double), cudaMemcpyHostToDevice));
    initMatrix(h_a, N, M, 1.0);
    initMatrix(h_b, N, M, 2.0);
    initMatrix(h_c, N, M, 0.0);
    assert(h_a[0] == 1.0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    triadBandwidth<<<grid, block>>>(dev_a, dev_b, dev_c, q, N, M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "kernel error launch %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpy(h_c, dev_c, N * M * sizeof(double), cudaMemcpyDeviceToHost));

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    // compute gpu_time from events ....
    double time_sec = 1.e-3 * gpu_time;
    double bytes = 3 * N * M * sizeof(double);
    double gbytes = 1.0e-9 * bytes/time_sec;
    printf("block size %i * %i\n",tile_width, tile_width);
    printf("CUDA gbytes = %10.2f , Time = %10.5f [s] dim=%d\n", gbytes, time_sec, N);
    // free on host
    free(h_a);
    free(h_b);
    free(h_c);
    // free on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
