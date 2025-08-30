## Compilers
CXX = mpicxx
NVCC = nvcc
CYY = g++

## Flags
CFLAGS = -O3 -fopenmp -march=native -std=c++17
CFLAGS_ = -O3 -march=native -std=c++17
CUDAFLAGS = -Xcompiler=-fopenmp
LDFLAGS = -lcudart -L${CUDA_HOME}/lib64 -fopenmp

## Targets
a4: main.o read_matrix.o mul_bsr.o 
	@$(CXX) $^ -o $@ $(LDFLAGS)

main.o: main.cpp
	@$(CXX) $(CFLAGS) -c $<

read_matrix.o: read_matrix.cpp
	@$(CYY) $(CFLAGS_) -c $<  # Compile read_matrix.cpp using g++

mul_bsr.o: mul_bsr.cu
	@$(NVCC) $(CUDAFLAGS) -c $< -o $@  # Compile mul_bsr.cu using nvcc

clean:
	@rm -f *.o a4

.PHONY: clean