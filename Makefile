CPU_SRC ?= model.c
CUDA_SRC ?= model.cu

CC = g++
CFLAGS = -march=native -ffast-math -O3 
LDFLAGS = -lm -fopenmp

NVCCFLAGS = -O3 --use_fast_math -Xcompiler -fopenmp
CUDA_LDFLAGS = -lcublas --expt-relaxed-constexpr

EXECUTABLE = executable

.SILENT:
	
run_cpu: $(CPU_SRC)
	$(CC) $(CFLAGS) $(CPU_SRC) -o $(EXECUTABLE) $(LDFLAGS)
	OMP_NUM_THREADS=8 ./$(EXECUTABLE)

run_cuda: $(CUDA_SRC)
	nvcc $(NVCCFLAGS) $(CUDA_SRC) -o $(EXECUTABLE) $(CUDA_LDFLAGS)
	OMP_NUM_THREADS=8 ./$(EXECUTABLE)

profile_cpu: $(CPU_SRC)
	$(CC) $(CFLAGS) -pg $(CPU_SRC) -o $(EXECUTABLE) $(LDFLAGS)
	OMP_NUM_THREADS=8 ./$(EXECUTABLE)
	gprof $(EXECUTABLE) gmon.out
	rm $(EXECUTABLE) gmon.out

profile_cuda: $(CUDA_SRC)
	nvcc $(NVCCFLAGS) -pg $(CUDA_SRC) -o $(EXECUTABLE) $(CUDA_LDFLAGS)
	OMP_NUM_THREADS=8 ./$(EXECUTABLE)
	gprof $(EXECUTABLE) gmon.out
	rm $(EXECUTABLE) gmon.out

clean:
	@echo "Cleaning up..."
	rm -f $(EXECUTABLE) gmon.out
	rm -f gpt2.o

.PHONY: run_cpu run_gpu profile_cpu profile_cuda clean