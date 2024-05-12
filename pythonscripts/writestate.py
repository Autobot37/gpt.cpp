import numpy as np
import torch
from gpt2 import *

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype="float32"):
    write_fun = write_fp32
    write_fun(model_tensors["transformer.wte.weight"], file)
    write_fun(model_tensors["transformer.wpe.weight"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L):
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file)
    write_fun(model_tensors["transformer.ln_f.bias"], file)

def write_model(model, filename):
    header = torch.zeros(8, dtype=torch.int32)
    header[0] = 3737
    header[1] = 1
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    
    params = {name : param.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as f:
        f.write(header.numpy().tobytes())
        write_tensors(params, model.config.n_layer, f)
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    pass

gpt = GPT.from_pretrained("gpt2")
params_path = "params.bin"
write_model(gpt, params_path)

