# GPT2 implementation in C and Cuda.
[Screencast from 15-05-24 11:47:23 PM IST.webm](https://github.com/Autobot37/gpt.cpp/assets/93463931/0763197f-a849-4fb0-b5bf-040b0d4835b7)

# Tasks did
Global structure is based on karpathy llm.c inference only implementation + additionaly

- Separate Optimized kernels for Cuda and Cpu.
- KV Cache.

# Todo
- Optimizing Attention_forward kernel.
- profiling and further optimizing based on profile.

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
