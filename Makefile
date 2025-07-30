CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
BUILD    := build

.PHONY: all test clean
all: $(BUILD)/test_ref

$(BUILD)/test_ref: tests/test_ref.cpp src/ref.cpp include/ck/ref.h tests/check.h | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_ref.cpp src/ref.cpp -o $@

$(BUILD):
	mkdir -p $@

test: $(BUILD)/test_ref
	./$(BUILD)/test_ref

clean:
	rm -rf $(BUILD)
