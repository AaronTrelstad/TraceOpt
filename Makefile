NVCC ?= nvcc
NVCC_FLAGS = -O2 -arch=sm_70

all: sync_benchmark

sync_benchmark: sync_benchmark.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f sync_benchmark

.PHONY: all clean
