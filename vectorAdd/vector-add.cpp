#include <mpi.h>
#include <cassert>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>

#define BLOCK_SIZE 32

#define MPI_CALL( call ) \
{   \
    int mpi_status = call; \
    if ( 0 != mpi_status ) \
    { \
        char mpi_error_string[MPI_MAX_ERROR_STRING]; \
        int mpi_error_string_length = 0; \
        MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
        if ( NULL != mpi_error_string ) \
            fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %s (%d).\n", #call, __LINE__, __FILE__, mpi_error_string, mpi_status); \
        else \
            fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %d.\n", #call, __LINE__, __FILE__, mpi_status); \
    } \
}

#define CUDA_CALL( X ) { \
    cudaError_t mycudaerror = ( X ) ; \
    if (mycudaerror != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s [%s:%d]\n", cudaGetErrorString(mycudaerror), __FILE__, __LINE__) ;\
    }}

void launch_vectorAdd_kernel(double *a, double *b, double *c, int n);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int local_rank, rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("rank = %d, size = %d\n", rank, size);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("num gpus is equal to %d\n", num_gpus);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    cudaSetDevice(local_rank%num_gpus);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("MPI rank %d , with local rank %d is using GPU %d on host %s\n", rank, local_rank, local_rank%num_gpus, hostname);
    double *h_A, *h_B, *h_C;
    int N = 10;
    assert(N % size == 0);
    int n = N / size;
    double *A, *B, *C;
    A = (double *) malloc(N * sizeof(double));
    B = (double *) malloc(N * sizeof(double));
    C = (double *) malloc(N * sizeof(double));
    h_A = (double *) malloc(n * sizeof(double));
    h_B = (double *) malloc(n * sizeof(double));
    h_C = (double *) malloc(n * sizeof(double));

    for(int i = 0; i < N; ++i)
    {
       A[i] = i;
       B[i] = N + i;
    }
    MPI_CALL(MPI_Scatter((void *)A, N / size, MPI_DOUBLE, (void *) h_A, N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Scatter((void *)B, N / size, MPI_DOUBLE, (void *) h_B, N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    for(int i = 0; i < n; ++i)
    {
        printf("MPI rank %d , h_A[%d] =  %f, h_B[%d] = %f\n", rank, n * rank + i, h_A[i], n * rank + i, h_B[i]);
        //printf("MPI rank %d , h_A[%d] =  %f, h_B[%d] = %f\n", rank, h_A[i], i, h_B[i]);
    }

    double *dev_A, *dev_B, *dev_C;
    CUDA_CALL(cudaMalloc(&dev_A, n * sizeof(double)));
    CUDA_CALL(cudaMalloc(&dev_B, n * sizeof(double)));
    CUDA_CALL(cudaMalloc(&dev_C, n * sizeof(double)));

    CUDA_CALL(cudaMemcpy(dev_A, h_A, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_B, h_B, n * sizeof(double), cudaMemcpyHostToDevice));


    launch_vectorAdd_kernel(dev_A, dev_B, dev_C, n);

    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(h_C, dev_C, n * sizeof(double), cudaMemcpyDeviceToHost));


    for(int i = 0; i < n; ++i)
    {
        printf("MPI rank %d , h_C[%d] = %f\n", rank, n * rank + i, h_C[i]);
    }

    MPI_CALL(MPI_Gather((void *) h_C, n, MPI_DOUBLE, (void *) C, n, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if (rank == 0)
    {
        for(int i = 0; i < N; ++i)
        {
            printf("C[%d] = %5.2f\n", i, C[i]);
        }
    }

    free(A); free(B); free(h_A); free(h_B); free(h_C);
    CUDA_CALL(cudaFree(dev_A));
    CUDA_CALL(cudaFree(dev_B));
    CUDA_CALL(cudaFree(dev_C));
    MPI_Finalize();
    return 0;
}
