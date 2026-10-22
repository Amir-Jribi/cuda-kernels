TOUBKAL_SUBMIT_CMD ?= srun --mpi=pmix
NP ?= 2
NVCC=nvcc
MPICXX=mpicxx
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM37    := -gencode arch=compute_37,code=sm_37
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM52    := -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_SM90    := -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
GENCODE_FLAGS   := $(GENCODE_SM90)
NVCC_FLAGS = -lineinfo $(GENCODE_FLAGS) -Xptxas --optimize-float-atomics -std=c++11
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -std=c++11
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart
vector-add: Makefile vector-add.cpp vector-add.o
    $(MPICXX) $(MPICXX_FLAGS) vector-add.cpp  vector-add.o $(LD_FLAGS) -o vector-add

vector-add.o: Makefile vector-add.cu
    $(NVCC) $(NVCC_FLAGS) vector-add.cu -c

.PHONY.: clean

clean:
    rm -f vector-add vector-add_kernels.o *.compute-sanitizer.log *.nsys-rep

sanitize: vector-add
    $(TOUBKAL_SUBMIT_CMD) -n $(NP) compute-sanitizer --log-file vector-add.%q{PMIX_RANK}.compute-sanitizer.log ./vector-add

run: vector-add
    $(TOUBKAL_SUBMIT_CMD) -n $(NP) ./vector-add

profile: vector-add
    $(TOUBKAL_SUBMIT_CMD) -n $(NP) nsys profile --trace=mpi,cuda,nvtx -o vector-add.%q{PMIX_RANK} ./vector-add
