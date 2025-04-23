# 
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcudart -lcusparse 

all: fused_kernel

fused_kernel: kernel.cu
	nvcc $(INC)  kernel.cu -o fused_kernel $(LIBS) -x=cu  -Xcompiler=-msse4

clean:
	rm -f fused_kernel

test:
	@echo "\n==== fused_kernel Test ====\n"
	./fused_kernel

.PHONY: clean all test
