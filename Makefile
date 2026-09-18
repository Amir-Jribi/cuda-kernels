CC = gcc
FC = nvfortran
NVCC = nvcc

CFLAG = -O3
FFLAG = -cuda -gpu=cc90 -fast
NVCCFLAG = -arch=sm_90 -Xcompiler=-O3

dummy:
        @echo -e "make saxpy\t\tcompile saxpy kernel"
        @echo -e "make stride\t\tcompile stride kernel"
        @echo -e "make clean\t\trm -f *.x *.mod"


saxpy:    saxpy.x
offset:    offset.x
stride:    stride.x


saxpy.x: saxpy.cu
        $(NVCC) $(NVCCFLAG) $< -o $@

offset.x: offset.cu
        $(NVCC) $(NVCCFLAG) $< -o $@

stride.x: stride.cu
        $(NVCC) $(NVCCFLAG) $< -o $@

clean:
        rm -f *.x *.mod *.ptx
