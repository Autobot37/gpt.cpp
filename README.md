# GPT2 implementation in C and Cuda.
[Screencast from 15-05-24 11:47:23 PM IST.webm](https://github.com/Autobot37/gpt.cpp/assets/93463931/0763197f-a849-4fb0-b5bf-040b0d4835b7)
# Tasks did
- Separate kernels for Cuda and Cpu.
- All kernels from scratch except matmul.
- tokenizer implementation.
- inference implementation.

# Todo
- Optimising Attention_forward kernel.
- profiling and further optimizing based on profile.

# To run
'''
git clone -b cuda https://github.com/Autobot37/gpt.cpp
python3 pythonscripts/prepare_tokenizer.py
python3 writestate.py
make run_cuda
'''

# Dependencies
- python tiktoken module.
- nvidia cuda toolkit.
