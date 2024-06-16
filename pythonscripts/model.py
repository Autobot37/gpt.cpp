import os
import math
from dataclasses import dataclass
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#-----------------------------------------------------------------------------
import tiktoken
from anytree import Node, RenderTree

enc = tiktoken.get_encoding("gpt2")
prompt = "am i dreamin is there more like us?"
tokens = enc.encode(prompt)
n = enc.n_vocab 

vocab = []
for i in range(n):
    vocab.append(enc.decode([i]))


class TrieNode:
    def __init__(self, char=None):
        self.children = {}
        self.token_length = -1
        self.token_id = -1
        self.char = char

    def init(self, strings):
        for i, token in enumerate(strings):
            node = self
            for char in token:
                if char not in node.children:
                    child_node = TrieNode(char)
                    node.children[char] = child_node
                node = node.children[char]
            node.token_length = len(token)
            node.token_id = i
    
    def encode(self, prompt):
        encoded = []
        node = self
        i = 0
        last_token_id = -1
        while i < len(prompt):
            key = prompt[i]
            if key not in node.children:
                print("Not found", key)
                node = self
                encoded.append(last_token_id)
                last_token_id = -1
                continue
            node = node.children[key]
            if node.token_id != -1:
                last_token_id = node.token_id
            i += 1
        encoded.append(last_token_id)
        return encoded
    
    def create_tree(self):
        self.node = Node(self.char)
        for char, child in self.children.items():
            child_node = child.create_tree()
            child_node.parent = self.node
        return self.node

    def print_tree(self):
        root_node = self.create_tree()
        print(RenderTree(root_node))            
#//-----------------------------------------------------------------------------

@dataclass
class Config:
    n_embd:int = 4
    n_head:int = 2
    block_size:int = 4
    vocab_size:int = 4
    n_layer:int = 2

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class NewGELU(nn.Module):
    def forward(self,input):
        out =  0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        return out

class Layernorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        std = (var).sqrt()
        rstd = 1.0 / (std + 1e-6)
        out = self.weight * (x - mean) * rstd + self.bias
        return out
        
class CausalSefAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head, C//self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = Layernorm(config.n_embd)
        self.attn = CausalSefAttention(config)
        self.ln_2 = Layernorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = Layernorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b,t = idx.size()
        assert t <= self.config.block_size , f"More sized"
        pos = torch.arange(0, t, dtype=torch.int, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        x = x.to(torch.float)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:,[-1],:])
            loss = None
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        if model_hf.config.n_layer == 12 and model_hf.config.n_head == 12 and model_hf.config.n_embd == 768:
            model_type = 'gpt2'
        elif model_hf.config.n_layer == 24 and model_hf.config.n_head == 16 and model_hf.config.n_embd == 1024:
            model_type = 'gpt2-medium'
        elif model_hf.config.n_layer == 36 and model_hf.config.n_head == 20 and model_hf.config.n_embd == 1280:
            model_type = 'gpt2-large'
        elif model_hf.config.n_layer == 48 and model_hf.config.n_head == 25 and model_hf.config.n_embd == 1600:
            model_type = 'gpt2-xl'
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}


        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 
        config_args['block_size'] = 1024 
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, check=False):
        assert max_new_tokens - idx.size(1) >0, "No new tokens to generate"
        for _ in range(max_new_tokens - idx.size(1)):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) if check is False else torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


#//-----------------------------------------------------------------------------

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

def write_tokenizer(enc, filename):
    n = enc.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n # number of tokens
    header[3] = enc.eot_token # EOT token
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length))  # Write the length as a 1-byte unsigned integer
            file.write(b)  # Write the actual bytes
    print(f"wrote {filename}")

def write_state(model, path):
    header = torch.zeros(8, dtype=torch.int32)
    header[0] = 3737
    header[1] = model.config.vocab_size
    header[2] = max_length = 128
    header[3] = num_tokens = 32

    tokens = torch.randint(0, model.config.vocab_size, (1, num_tokens), dtype=torch.int32)
    next_tokens = model.generate(tokens, max_length, check=True)[0].to(torch.int32)
    print(next_tokens.shape)
    print(tokens.shape)
    next_tokens = next_tokens.cpu().numpy()
    tokens = tokens[0].cpu().numpy()
    print(next_tokens)
    print(tokens)
    with open(path, "wb") as f:
        f.write(header.numpy().tobytes())
        f.write(tokens.tobytes())
        f.write(next_tokens.tobytes())
    print(f"wrote {path}")

device = "cpu" if not torch.cuda.is_available() else "cuda"

import tiktoken
enc = tiktoken.get_encoding("gpt2")
write_tokenizer(enc, "tokenizer.bin")

model = GPT.from_pretrained('singhshiva/kaggle')
params_path = "params.bin"
write_model(model, params_path)

debug_path = "debug.bin"
write_state(model, debug_path)

#//-----------------------------------------------------------------------------
model = model.to(device)
encoded = enc.encode("Hello anon are you")
print(encoded)
tokens = torch.tensor(encoded, dtype=torch.int32).unsqueeze(0).to(device)
gen = model.generate(tokens, max_new_tokens=128, temperature=0.75)
print(gen)
decoded = enc.decode(gen[0].cpu().numpy())
print(decoded)