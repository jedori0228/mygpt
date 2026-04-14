import tiktoken
import numpy as np

with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

# Split train and val

nchar = len(data)

train_max_index = int(0.9*nchar)
data_train = data[:train_max_index]
data_val = data[train_max_index:]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")

tokens_train = enc.encode_ordinary(data_train)
tokens_val = enc.encode_ordinary(data_val)

print(f"train has {len(tokens_train):,} tokens")
print(f"val has {len(tokens_val):,} tokens")

# export to bin files
train_ids = np.array(tokens_train, dtype=np.uint16)
val_ids = np.array(tokens_val, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')

