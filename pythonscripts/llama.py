from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLaMAModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None

@dataclass
class ModelArgs:
  dim : int = 4*4*4
  n_layers : int = 4
  n_heads : int = 4*4
  n_kv_heads : int = 4
  vocab_size : int = 128
  multiple_of : int = 256
  ffn_dim_multiplier: Optional[float] = 64
  norm_eps : float = 1e-5
  max_batch_size : int = 32
  max_seq_len : int = 2048

class RMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
  def forward(self, x):
    x = x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True) + self.eps)*self.weight
    return x

def pre(head_dim, seq_len):
  theta = torch.arange(0, head_dim, 2).float()
  theta = 1.0/(10000**(theta/head_dim))
  m = torch.arange(seq_len)
  freqs = torch.outer(m, theta).float()
  freqs_complex = torch.polar(torch.ones_like(freqs),freqs)
  return freqs_complex
###pre gives complex for [x1,x2,x3...xseq_len]

def apply_embs(x, freqs_complex):
  #x- shape -> b,1(seq_len),n_h,head_dim
  x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
  freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
  x_rotated = x_complex * freqs_complex
  x_out = torch.view_as_real(x_rotated)
  x_out = x_out.reshape(*x.shape)
  return x_out.type_as(x)

def repeat_kv(x,n_rep):
  b,t,kv_heads,head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:,:,:,None,:]
      .expand(b,t,kv_heads,n_rep,head_dim)
      .reshape(b,t,kv_heads*n_rep,head_dim)
  )

class Attention(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.n_kv_heads = args.n_kv_heads
    self.n_heads_q = args.n_heads
    self.n_rep = self.n_heads_q // self.n_kv_heads
    self.head_dim = args.dim // args.n_heads

    self.wq = nn.Linear(args.dim,args.n_heads*self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_kv_heads*self.head_dim,bias=False)
    self.wv = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)

    self.cache_k = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads, self.head_dim))
    self.cache_v = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads, self.head_dim))

  def forward(self, x, start_pos, freqs_complex):
    #x = b,1,dim
    b,t,_ = x.shape
    xq = self.wq(x)
    xk = self.wk(x)
    xv = self.wv(x)

    xq = xq.view(b,t,self.n_heads_q, self.head_dim)
    xk = xk.view(b,t,self.n_kv_heads, self.head_dim)
    xv = xv.view(b,t,self.n_kv_heads, self.head_dim)

    xq = apply_embs(xq, freqs_complex)
    xk = apply_embs(xk, freqs_complex)

    self.cache_k[:b,start_pos:start_pos+t] = xk
    self.cache_v[:b,start_pos:start_pos+t] = xv

    keys = self.cache_k[:b,:start_pos+t]
    values = self.cache_v[:b,:start_pos+t]
    #KEYS = b,seq_len_kv,h_kv,head_dim
    keys = repeat_kv(keys, self.n_rep)
   #Keys = b,seq_len_kv,h_q,head_dim
    values = repeat_kv(values, self.n_rep)

    xq = xq.transpose(1,2)
    keys = keys.transpose(1,2)
    values = values.transpose(1,2)

    scores = torch.matmul(xq, keys.transpose(2,3))/math.sqrt(self.head_dim)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values)
    output = output.transpose(1,2).contiguous().view(b, t , -1)

    return self.wo(output)

class FeedForward(nn.Module):
  def __init__(self,args):
    super().__init__()
    hidden_dim = 4*args.dim
    hidden_dim = int(2*hidden_dim/3)
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1)//args.multiple_of)

    self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
    self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

  def forward(self,x):
    swish = F.silu(self.w1(x))
    x_V = self.w3(x)
    x = swish*x_V
    x = self.w2(x)
    return x

class Encoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads

    self.att = Attention(args)
    self.ff = FeedForward(args)

    self.att_norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.ff_norm = RMSNorm(args.dim, eps=args.norm_eps)

  def forward(self, x, start_pos, freqs_complex):
    h = x + self.att.forward(
        self.att_norm(x), start_pos, freqs_complex
    )
    out = h + self.ff(self.ff_norm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, args):
    super().__init__()

    assert args.vocab_size != -1

    self.args = args
    self.vocab_size = args.vocab_size
    self.n_layers = args.n_layers
    self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

    self.layers = nn.ModuleList()
    for id in range(args.n_layers):
      self.layers.append(Encoder(args))

    self.norm = RMSNorm(args.dim)
    self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    self.freqs_complex = pre(self.args.dim//self.args.n_heads,self.args.max_seq_len*2)
    print(f"you have {self.num_params()} params , handle them")

  def num_params(self):
    return sum(p.numel() for p in self.parameters())

  def forward(self, tokens, start_pos):
    b,t = tokens.shape
    assert t == 1, "Only 1 token boss"

    h = self.tok_embeddings(tokens)
    freqs_complex = self.freqs_complex[start_pos:start_pos+t]

    for l in self.layers:
      h = l(h, start_pos, freqs_complex)
    h = self.norm(h)
    output = self.output(h).float()
    return output

# args = ModelArgs
# args.vocab_size = 5090
# model = Transformer(args)
# x = torch.ones((1,4),dtype=torch.long)
# for i in range(3):
#     print(model(x[:,i:i+1],i).detach())