# GPT2 implementation in C and Cuda.

[Screencast from 17-06-24 05:30:50 PM IST.webm](https://github.com/Autobot37/gpt.cpp/assets/93463931/330b4364-e4ae-467e-842f-085b0aaf8355)

# Tasks did
Global structure is based on karpathy llm.c inference only implementation + additionaly

- Separate Optimized kernels for Cuda and Cpu.
- KV Cache.
- Byte Pair Encoding with Trie.
- Python, C and CUDA test to ensure same output between huggingface - custommodel - c file - CUDA file.
- Multinomial Sampling.

# Todo
- cudaSgemmBatchedStrided is bottleneck consuming ~84% time so maybe finding alt to it.

# To run GPU version
```bash
git clone -b master https://github.com/Autobot37/gpt.cpp
pip install -r requirements.txt
python3 pythonscripts/model.py
make run_cuda
```

# To run CPU version
```bash
git clone -b master https://github.com/Autobot37/gpt.cpp
pip install -r requirements.txt
python3 pythonscripts/model.py
make run_cpu
```

# System Tested
- Ubuntu 22.04
- Cuda 11.8

