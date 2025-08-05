CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
# Kernels and benchmarks build with -O3 and native tuning; tests keep -O2 portability.
FAST     := -O3 -march=native
UNAME    := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  # Apple's BLAS, used only as a reference ceiling in the benchmark table.
  BENCH_EXTRA := -DCK_HAVE_ACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  FAST := -O3 -mcpu=native
endif
BUILD    := build
PYTHON   ?= python3
THREADS  ?= 4

.PHONY: all test bench microbench roofline-host report cuda cuda-test cuda-syntax clean
all: $(BUILD)/test_ref $(BUILD)/test_rowops_math $(BUILD)/test_cpu_gemm $(BUILD)/bench_cpu_gemm $(BUILD)/microbench

$(BUILD)/test_ref: tests/test_ref.cpp src/ref.cpp include/ck/ref.h tests/check.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_ref.cpp src/ref.cpp -o $@

$(BUILD)/test_rowops_math: tests/test_rowops_math.cpp cuda/rowops_math.h src/ref.cpp include/ck/ref.h tests/check.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_rowops_math.cpp src/ref.cpp -o $@

$(BUILD)/test_cpu_gemm: tests/test_cpu_gemm.cpp src/cpu_gemm.cpp src/ref.cpp include/ck/*.h tests/check.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(FAST) tests/test_cpu_gemm.cpp src/cpu_gemm.cpp src/ref.cpp -o $@ -lpthread

$(BUILD)/bench_cpu_gemm: bench/bench_cpu_gemm.cpp src/cpu_gemm.cpp include/ck/cpu_gemm.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(FAST) bench/bench_cpu_gemm.cpp src/cpu_gemm.cpp -o $@ -lpthread $(BENCH_EXTRA)

$(BUILD)/microbench: bench/microbench.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(FAST) bench/microbench.cpp -o $@ -lpthread

$(BUILD):
	mkdir -p $@

test: $(BUILD)/test_ref $(BUILD)/test_rowops_math $(BUILD)/test_cpu_gemm
	./$(BUILD)/test_ref
	./$(BUILD)/test_rowops_math
	./$(BUILD)/test_cpu_gemm
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

bench: $(BUILD)/bench_cpu_gemm
	./$(BUILD)/bench_cpu_gemm

microbench: $(BUILD)/microbench
	./$(BUILD)/microbench $(THREADS)

# Measure this machine and add it to roofline/devices.json under HOST_KEY.
HOST_KEY  ?= host
HOST_NAME ?= host CPU (measured)
roofline-host: $(BUILD)/microbench
	./$(BUILD)/microbench $(THREADS) > results/$(HOST_KEY)_microbench.txt
	$(PYTHON) -m roofline.host results/$(HOST_KEY)_microbench.txt $(HOST_KEY) "$(HOST_NAME)"

report:
	$(PYTHON) -m roofline.report

# ---- CUDA (needs nvcc and an NVIDIA GPU) ----------------------------------------------
NVCC      ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -lineinfo -Iinclude
CUDA_LIB  := cuda/sgemm.cu cuda/rowops.cu
CUDA_HDRS := $(wildcard cuda/*.cuh cuda/*.h) include/ck/ref.h tests/check.h

cuda: $(BUILD)/test_sgemm $(BUILD)/test_rowops

$(BUILD)/test_sgemm: cuda/test_sgemm.cu $(CUDA_LIB) src/ref.cpp $(CUDA_HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) cuda/test_sgemm.cu $(CUDA_LIB) src/ref.cpp -o $@

$(BUILD)/test_rowops: cuda/test_rowops.cu $(CUDA_LIB) src/ref.cpp $(CUDA_HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) cuda/test_rowops.cu $(CUDA_LIB) src/ref.cpp -o $@

cuda-test: cuda
	./$(BUILD)/test_sgemm
	./$(BUILD)/test_rowops

# Static check of every .cu file with clang's CUDA front end and a stub runtime header
# (force-included first, because clang's CUDA std wrappers need __host__ defined).
# Works without the CUDA toolkit. It catches syntax and type errors only; it does not
# generate GPU code and proves nothing about correctness or speed.
CLANG ?= clang++
cuda-syntax:
	@for f in cuda/*.cu; do \
	  echo "clang -fsyntax-only $$f"; \
	  $(CLANG) -x cuda --cuda-gpu-arch=sm_80 -nocudainc -nocudalib -std=c++17 \
	    -Iinclude -Itools/cuda_syntax -include cuda_runtime.h -fsyntax-only $$f || exit 1; \
	done

clean:
	rm -rf $(BUILD)
