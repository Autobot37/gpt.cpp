# gpt.cpp
Correct gpt2 implementation in C.

current benchmarks on 11thgencorei5:
    block_size = 128;
	vocab_size = 8092;
	n_embd = 128;
	n_head = 4;
	n_layer = 2;

total_size:2482432bytes[float32]

takes 2.7 seconds with unoptimised[openmp 8 threads] matrix multiplication.




## Install

```bash
git clone https://github.com/Autobot37/gpt.cpp
cd gpt.cpp
g++ -ffast-math gpt.cpp -o run
./run
```

