#include <stdio.h>

#define CUDA_CHECK( X ) { \
    cudaError_t mycudaerror = ( X ) ; \
    if (mycudaerror != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s [%s:%d]\n", cudaGetErrorString(mycudaerror), __FILE__, __LINE__) ;\
    }}
void initMatrix(float *a, int N, float num)
{
    for(int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}
__global__ void offsetCopy(int N, float *odata, float *idata, int offset)
{
    int xid = (blockDim.x * blockIdx.x + threadIdx.x) + offset;
    if (xid < N) odata[xid] = idata[xid];
}
int main(int argc, char **argv)
{
    int offset;
    int N = 16776960;
    float q = 6.0;
    if(argc==2)
    {
        offset=atoi(argv[1]);
    }
    else if (argc != 2)
    {
        fprintf(stderr,"Usage: %s [offset]\n",argv[0]);
        exit(1);
    }


    size_t bytes = N * sizeof(float);
    printf("Using array dim=%d [size=%d (bytes)]\n", N, bytes);


    float *out, *in;

    // allocate mem on host,
    out = (float *) malloc(N * sizeof(float));
    in = (float *) malloc(N * sizeof(float));
    initMatrix(in, N, q);


    float *dev_out, *dev_in;
    CUDA_CHECK(cudaMalloc(&dev_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_in, N * sizeof(float)));


    CUDA_CHECK(cudaMemcpy(dev_in, in, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    offsetCopy<<<grid, block>>>(N, dev_out, dev_in, offset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "error in kernel launch %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;

    cudaEventElapsedTime(&gpu_time, start, stop);

    double time_sec = 1.e-3 * gpu_time;
    double trans_bytes = 2 * (N - offset) * sizeof(float); // 1 store + 1 load
    double gbytes = 1.0e-9 * trans_bytes/time_sec;
   // printf("block size %i * %i\n",tile_width, tile_width);
    printf("Bandwidth GB/s = %10.2f , Time = %10.5f [s] dim=%d\n", gbytes, time_sec, N);

    CUDA_CHECK(cudaMemcpy(out, dev_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < N; ++i)
    {
        if((i + offset) < N)
        {
            if (out[i + offset] != q)
            {
                fprintf(stderr, "Element at [%d] is not equal to %d\n", i+offset, q);
            }
        }
    }


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_in);
    cudaFree(dev_out);

    free(in);
    free(out);
    return 0;
}
