from transformers import AutoModelForCausalLM
from pythonscripts.model import GPT   
import torch
import tiktoken

device = "cpu" if not torch.cuda.is_available() else "cuda"

repo = "gpt2"
tokenizer = tiktoken.get_encoding("gpt2")
prompt = "hey anon"
encoded = tokenizer.encode(prompt)
encoded = torch.tensor(encoded, dtype=torch.int64).unsqueeze(0).to(device)

model_hf = AutoModelForCausalLM.from_pretrained(repo).to(device)
model_custom = GPT.from_pretrained('gpt2').to(device)

tokens_hf = model_hf.generate(encoded, max_length=128, do_sample=False)
tokens_custom = model_custom.generate(encoded, 128, temperature=1.0, check=True)
print("---------")
print(tokens_hf.shape, tokens_hf)
print(tokens_custom.shape, tokens_custom)

assert torch.all(tokens_hf == tokens_custom), "Models do not match" 
print("All tests passed!")