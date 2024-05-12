CPU_SRC ?= gpt2.c
CUDA_SRC ?= gpt2.cu

CFLAGS = -march=native -lm --ffast-math -O3 -funroll-loops
NVCCFLAGS = -O3 --use_fast_math -Xcompiler -fopenmp
LDFLAGS = -lcublas -lcublasLt -lm

EXECUTABLE = executable

.SILENT: run_cpu run_gpu profile_cpu profile_cuda clean
.PHONY: run_cpu run_gpu profile_cpu profile_cuda clean

run_cpu: $(CPU_SRC)
	$(CC) $(CFLAGS) $(CPU_SRC) -o $(EXECUTABLE)
	OMP_NUM_THREADS=8 ./$(EXECUTABLE)
	rm $(EXECUTABLE)

run_cuda: $(CUDA_SRC)
	nvcc $(NVCCFLAGS) $(CUDA_SRC) -o $(EXECUTABLE) $(LDFLAGS)
	OMP_NUM_THREADS=8 ./$(EXECUTABLE)
	rm $(EXECUTABLE)

profile_cpu: $(CPU_SRC)
	$(CC) $(CFLAGS) -pg $(CPU_SRC) -o $(EXECUTABLE)
	OMP_NUM_THREADS=8 ./$(EXECUTABLE)
	gprof $(EXECUTABLE) gmon.out
	rm $(EXECUTABLE) gmon.out

profile_cuda: $(CUDA_SRC)
	nvcc $(NVCCFLAGS) -pg $(CUDA_SRC) -o $(EXECUTABLE) $(LDFLAGS)
	OMP_NUM_THREADS=8 ./$(EXECUTABLE)
	gprof $(EXECUTABLE) gmon.out
	rm $(EXECUTABLE) gmon.out

clean:
	@echo "Cleaning up..."
	rm -f $(EXECUTABLE) gmon.out
	rm -f gpt2.o