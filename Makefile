CC = gcc
NVCC = nvcc
CFLAGS = -std=c99
LDFLAGS = -lm -pg
TARGET_CPU = gpt2_cpu
TARGET_CUDA = gpt2_cuda
PROFILE_TARGET_CPU = gpt2_profile_cpu
PROFILE_TARGET_CUDA = gpt2_profile_cuda

SRCS_C = gpt2.c
SRCS_CUDA = gpt2.cu
OBJS_C = $(SRCS_C:.c=.o)
OBJS_CUDA = $(SRCS_CUDA:.cu=.o)

.SILENT:
.PHONY: all clean run_cpu run_cuda profile_cpu profile_cuda

all: $(TARGET_CPU) $(TARGET_CUDA)

$(TARGET_CPU): $(OBJS_C)
	$(CC) $(CFLAGS) $(OBJS_C) -o $@ $(LDFLAGS)

$(TARGET_CUDA): $(OBJS_CUDA)
	$(NVCC) $(OBJS_CUDA) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -c $< -o $@

run_cpu: $(TARGET_CPU)
	./$(TARGET_CPU)
	rm -f $(TARGET_CPU)

run_cuda: $(TARGET_CUDA)
	./$(TARGET_CUDA)
	rm -f $(TARGET_CUDA)

profile_cpu: $(PROFILE_TARGET_CPU)
	./$(PROFILE_TARGET_CPU)
	gprof $(PROFILE_TARGET_CPU)
	rm -f $(PROFILE_TARGET_CPU)

profile_cuda: $(PROFILE_TARGET_CUDA)
	./$(PROFILE_TARGET_CUDA)
	gprof $(PROFILE_TARGET_CUDA)
	rm -f $(PROFILE_TARGET_CUDA)

$(PROFILE_TARGET_CPU): $(OBJS_C)
	$(CC) $(CFLAGS) $(OBJS_C) -o $@ $(LDFLAGS)

$(PROFILE_TARGET_CUDA): $(OBJS_CUDA)
	$(NVCC) $(OBJS_CUDA) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS_C) $(OBJS_CUDA) $(TARGET_CPU) $(TARGET_CUDA) $(PROFILE_TARGET_CPU) $(PROFILE_TARGET_CUDA)
