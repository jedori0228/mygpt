import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

InputBaseDir = '/exp/dune/data/users/jskim/ml'
DataBaseDir = f'{InputBaseDir}'
TokenizerBasedir = f'{InputBaseDir}/tokenizer'

DataName = 'openwebtext'

# --- Tokenizer selection ---
# TokenizerType = 'gpt2'
TokenizerType = 'EnKoMix'

if TokenizerType == 'gpt2':
    import tiktoken
    enc_model = tiktoken.get_encoding("gpt2")
    # Define a wrapper to match the interface
    encode_func = lambda text: enc_model.encode_ordinary(text)
    eot_token_id = enc_model.eot_token
elif TokenizerType == 'EnKoMix':
    from tokenizers import Tokenizer
    custom_tokenizer = Tokenizer.from_file(f"{TokenizerBasedir}/EnKoMix/tokenizer.json")
    # HuggingFace tokenizers return an object; we need the .ids attribute
    encode_func = lambda text: custom_tokenizer.encode(text).ids
    # Get ID for </s> (or whichever you set as EOT)
    eot_token_id = custom_tokenizer.token_to_id("</s>")

# --- Dataset Loading ---
dataset = load_dataset("openwebtext", split="train", streaming=True)
total_rows = dataset.info.splits["train"].num_examples

test_size = 0.0005
val_size = int(total_rows * test_size)
print(f"Total rows in dataset: {total_rows}")
print(f"Calculated validation size: {val_size}")

seed = 2357
shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=10000)

val_stream = shuffled_dataset.take(val_size)
train_stream = shuffled_dataset.skip(val_size)

streams = {'val': val_stream, 'train': train_stream}

# --- Processing Loop ---
for split, stream in streams.items():
    output_path = os.path.join(DataBaseDir, DataName)
    os.makedirs(output_path, exist_ok=True)
    filename = f'{output_path}/{TokenizerType}/{split}.bin'
    
    pbar_total = val_size if split == 'val' else (total_rows - val_size)
    
    with open(filename, 'wb') as f:
        pbar = tqdm(desc=f"Writing {filename}", total=pbar_total, unit="docs")
        for example in stream:
            # Use the unified encode_func
            ids = encode_func(example['text'])
            
            # Append EOT token if it exists
            if eot_token_id is not None:
                ids.append(eot_token_id)
            
            # Convert to uint16 (fits vocab size up to 65535)
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            pbar.update(1)
        pbar.close()
