CUDA_PATH = /usr/local/cuda
TRT_PATH = /usr/lib/x86_64-linux-gnu
NVCC = $(CUDA_PATH)/bin/nvcc
CC = /usr/bin/g++
USE_FP16 = 0
CUFLAG = -w -std=c++14 -O3 -use_fast_math -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -DNDEBUG -DDEBUG_ENABLE=0 -DUSE_FP16=$(USE_FP16) -Xcompiler -fPIC
CPPFLAG = -w -std=c++14 -O3 -use_fast_math -fPIC
SOFLAG = $(CUFLAG) -shared
INCLUDE = -I. -I$(CUDA_PATH)/include
LDFLAG = -L$(CUDA_PATH)/lib64 -lcudnn -lcublas -lcublasLt -lcudart -L$(TRT_PATH)/lib -lnvinfer -lz

CU_SRC = ./GeluPlugin.cu
CU_OBJ = $(CU_SRC:%.cu=%.o)
CC_SRC = $(shell find ./ -name *.cc)
CC_OBJ = $(CC_SRC:%.cc=2.o)

all: Gelu.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

Gelu.so: $(CU_OBJ)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^
	

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt

.PHONY: test
test:
	make clean
	make all