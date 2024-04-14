import os
import requests
import tqdm
import tiktoken
import numpy as np
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
enc = tiktoken.get_encoding("gpt2")
encode = lambda s : enc.encode_ordinary(s)

import requests
from tqdm import tqdm

def download_file(url, fname, chunk_size=8192):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as f, tqdm(
        desc=fname,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)

DATA_DIR = "data"

def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_path = os.path.join(DATA_DIR, "stories.tar.gz")
    if not os.path.exists(data_path):
        download_file(data_url, data_path)
        print("downloaded")    
    else:
      print("already downloaded")
    
    
    data_dir = os.path.join(DATA_DIR, "stories")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        os.system(f"tar -xzf {data_path} -C {data_dir}")
        print("extracted")
    else:
        print("already extracted")

    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

def process_shard(shard_index, shard_filename):
    with open(shard_filename, "rb") as f:
        data = json.load(f)
    eot = enc._special_tokens['<|endoftext|>']
    rng = random.Random(3137 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example["story"]
        text = text.strip()
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    return all_tokens


def tokenize():
    data_dir = os.path.join(DATA_DIR, "stories")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:5]
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:
        print(f"Tokenizing {split_name}")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard, shard_index, shard_filename) 
                       for shard_index, shard_filename in enumerate(split_shards)]
            for future in as_completed(futures):
                all_tokens.extend(future.result())
        
        all_tokens_np = np.array(all_tokens)
        split_filename = os.path.join(DATA_DIR, f"stories_{split_name}.bin")
        with open(split_filename, "wb") as f:
            f.write(all_tokens_np.tobytes())
        print(f"{split_filename} saved")


if __name__ == "__main__":
    download()
    tokenize()