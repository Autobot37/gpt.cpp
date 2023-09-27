# gpt.cpp
Correct gpt2 implementation in C.

current benchmarks on Ubuntu20.04[VM] 6gb intel i5core  :
   	model.config.block_size = 128
	model.config.vocab_size = 4*8092
	model.config.n_embd = 128
	model.config.n_head = 4
	model.config.n_layer = 2


total_size:8000000bytes[float32]

takes 1.04 seconds with unoptimised[openmp 8 threads] matrix multiplication.


#Goals
full chat compatibility
as much as  low level control of hardware
doing quantization
more perfomance more optimization and more speed also more accurate  
## Install

```bash
git clone https://github.com/Autobot37/gpt.cpp
cd gpt.cpp
g++ -ffast-math -funsafe-loop-optimizations  -O3 gpt2.cpp -o run -lm -lstdc++
./run
```

