import os
import struct
import argparse
from typing import List
import torch
print(torch.cuda.is_available())
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

MODEL = "tokenizer.model"

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else MODEL
        self.sp_model = SentencePieceProcessor(model_file = model_path)
        self.model_path = model_path

        self.vocab_size = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()

        
    def encode(self,s:str, bos:bool, eos:bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        
        return t
    
    def decode(self, t:List[int]) -> str:
        return self.sp_model.decode(t)
    
    def export(self):

        tokens = []
        scores = []

        for i in range(self.vocab_size):

            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('_',' ')
            b = t.encode('utf-8')

            tokens.append(b)
            scores.append(s)
            

        max_token_len = max(len(t) for t in tokens)

        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_len))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)
        

   
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-t","--tokensizer_model",type=str,help="path to tokenizer model")
#     args = parser.parse_args()

#     t = Tokenizer(args.tokenizer_model)
#     t.export()