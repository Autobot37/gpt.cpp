import time
import json

class LLaMA:
  @staticmethod
  def build(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob('*.pth'))
    if len(checkpoints) == 0:print("no checkpoint in this dir")
    ckpt_path = checkpoints[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir)/"params.json", "r") as f:
      params = json.loads(f.read())
    
    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )

    tokenizer = Tokenizer(tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    print(f"loaded in {time.time() - start_time:.2f} seconds")

    return LLaMA(model, tokenizer)
  
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer
  
  @torch.inference_mode()
  def generate(self, prompt_tokens,max_gen_len,temp,top_p):
    pass
