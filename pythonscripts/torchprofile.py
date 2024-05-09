from gpt2 import *
from torch.profiler import profile, record_function, ProfilerActivity

x = torch.randint(10,(4,8,16))
gpt = GPT(Config)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    with record_function("model_inference"):
        gpt(x[0,:,:])

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))