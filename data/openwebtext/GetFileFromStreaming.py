# %%
import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

# %%
DataBaseDir = '/Users/jskim/Documents/MLStudy/mygpt/data'

# %%
DataName = 'openwebtext'

# %%
enc = tiktoken.get_encoding("gpt2")

# %%
dataset = load_dataset("openwebtext", split="train", streaming=True)

# %%
total_rows = dataset.info.splits["train"].num_examples

test_size = 0.0005
val_size = int(total_rows * test_size)
print(f"Total rows in dataset: {total_rows}")
print(f"Calculated validation size: {val_size}")

# %%
seed = 2357
shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=10000)

# %%
val_stream = shuffled_dataset.take(val_size)
train_stream = shuffled_dataset.skip(val_size)

# %%
streams = {'val': val_stream, 'train': train_stream}

# %%
def process_stream(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return ids

# %%
for split, stream in streams.items():
    # Points to the same directory as your script
    filename = f'{DataBaseDir}/{DataName}/{split}.bin'
    
    # Determine total for the progress bar
    # We know val_size, but for train, it's the remaining rows
    pbar_total = val_size if split == 'val' else (total_rows - val_size)
    
    with open(filename, 'wb') as f:
        pbar = tqdm(desc=f"Writing {filename}", total=pbar_total, unit="docs")
        for example in stream:
            # Process text to tokens
            ids = enc.encode_ordinary(example['text'])
            ids.append(enc.eot_token)
            
            # Convert to uint16 (fits gpt2 max_token_value 50256)
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            pbar.update(1)
        pbar.close()

# %%



