import sys
import pickle
import struct
import tiktoken
import torch

enc = tiktoken.get_encoding("gpt2")

def write():
    with open('dictionary.bin', 'wb') as f:
        magic_num = 1337
        n = enc.max_token_value + 1
        header = torch.zeros(8, dtype=torch.int32)
        header[0] = magic_num
        header[2] = n
        header[3] = enc.eot_token

        buffer = header.numpy().tobytes()
        f.write(buffer)

        for i in range(n):
            key = enc.decode_bytes([i])
            size_key = len(key)
            sizebuffer = struct.pack('B', size_key)
            f.write(sizebuffer)
            f.write(key)
    print("Dictionary written to binary file successfully.")

write()