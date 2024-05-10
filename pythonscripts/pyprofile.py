from line_profiler import LineProfiler
from gpt2 import *

x = torch.randint(10,(4,8,16))

mlp = MLP(Config)
attn = CausalSefAttention(Config)
blk = Block(Config)
gpt = GPT(Config)

profiler = LineProfiler()
profiler.add_function(attn.forward)
profiler.add_function(mlp.forward)
profiler.add_function(blk.forward)
profiler.add_function(gpt.forward)
profiler.add_function(gpt.generate)

profiler.enable_by_count()

attn(x.float()).shape
mlp(x.float()).shape
blk(x.float()).shape
gpt(x[0,:,:])
gpt.generate(x[0,:,:],max_new_tokens=32)

profiler.disable_by_count()
profiler.print_stats()
