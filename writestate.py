import numpy as np
import torch
from gpt2 import *

def write_fp32(file_path, tensor):
    with open(file_path, 'ab') as f:  
        float32_data = tensor.detach().numpy().astype(np.float32)
        f.write(float32_data.tobytes())

def write_tensors(model, filename):
    for name, params in model.named_parameters():
        data = params.data.cpu().numpy()
        write_fp32(filename, params)

def write_model(model, filename):
    header = torch.zeros(8, dtype=torch.int32)
    header[0] = 3737
    header[1] = 1
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        write_tensors(model, filename)
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    pass

gpt = GPT(Config)
params_path = "params.bin"
write_model(gpt, params_path)