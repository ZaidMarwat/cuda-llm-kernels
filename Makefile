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

.PHONY: all test bench clean
all: $(BUILD)/test_ref $(BUILD)/test_cpu_gemm $(BUILD)/bench_cpu_gemm

$(BUILD)/test_ref: tests/test_ref.cpp src/ref.cpp include/ck/ref.h tests/check.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_ref.cpp src/ref.cpp -o $@

$(BUILD)/test_cpu_gemm: tests/test_cpu_gemm.cpp src/cpu_gemm.cpp src/ref.cpp include/ck/*.h tests/check.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(FAST) tests/test_cpu_gemm.cpp src/cpu_gemm.cpp src/ref.cpp -o $@ -lpthread

$(BUILD)/bench_cpu_gemm: bench/bench_cpu_gemm.cpp src/cpu_gemm.cpp include/ck/cpu_gemm.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(FAST) bench/bench_cpu_gemm.cpp src/cpu_gemm.cpp -o $@ -lpthread $(BENCH_EXTRA)

$(BUILD):
	mkdir -p $@

test: $(BUILD)/test_ref $(BUILD)/test_cpu_gemm
	./$(BUILD)/test_ref
	./$(BUILD)/test_cpu_gemm

bench: $(BUILD)/bench_cpu_gemm
	./$(BUILD)/bench_cpu_gemm

clean:
	rm -rf $(BUILD)
