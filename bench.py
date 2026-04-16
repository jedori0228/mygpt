"""
generate.py — Load a checkpoint and generate text from a prompt.

Usage:
    python generate.py --checkpoint logs/ckpt_01000.pt --prompt "Hello" --max_tokens 100
    python generate.py --checkpoint auto --prompt "Once upon a time" --max_tokens 200 --seed 42
"""

import os
import glob
import argparse
import tqdm

import torch

from model import ModelConfig, MyGPT
from utils.load_model import load_model
from utils.hellaswag_helper import hellaswag_helper

MODEL_BAESDIR = os.environ['MODEL_BASEDIR']
DATA_BASEDIR = os.environ['DATA_BASEDIR']
TOKENIZER_BASEDIR = os.environ['TOKENIZER_BASEDIR']

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text from a MyGPT checkpoint")
    parser.add_argument('--checkpoint', type=str, default='auto',
                        help="Path to checkpoint file")
    args = parser.parse_args()

    # --- Device ---
    device = (torch.accelerator.current_accelerator().type
              if torch.accelerator.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Tokenizer ---
    #TokenizerType = 'gpt2'
    TokenizerType = 'EnKoMix'
    if TokenizerType == 'gpt2':
        import tiktoken
        enc_model = tiktoken.get_encoding("gpt2")
        encode_func = lambda text: enc_model.encode_ordinary(text)
        decode_func = lambda ids: enc_model.decode(ids)
        NVocab = enc_model.n_vocab
    elif TokenizerType == 'EnKoMix':
        from tokenizers import Tokenizer
        custom_tokenizer = Tokenizer.from_file(f"{TOKENIZER_BASEDIR}/EnKoMix/tokenizer.json")
        # HuggingFace tokenizers return an object; we need the .ids attribute
        encode_func = lambda text: custom_tokenizer.encode(text).ids
        decode_func = lambda ids: custom_tokenizer.decode(ids)
        NVocab = custom_tokenizer.get_vocab_size()

    # --- Load model ---
    ckpt_path = args.checkpoint
    model     = load_model(ckpt_path, device)

    # --- hellaswah ---
    HellaSwagRawInputBaseDir = f"{DATA_BASEDIR}/hellaswag"
    _hellaswag_helper = hellaswag_helper(HellaSwagRawInputBaseDir, encode_func)


    num_correct_norm = 0
    num_total = 0
    for i, example in tqdm.tqdm(enumerate(_hellaswag_helper.iterate_examples("val")), total=_hellaswag_helper.TotalExample['val'], desc='Hellaswag validation examples'):
        # render the example into tokens and labels
        _, tokens, mask, label = _hellaswag_helper.render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            logits = model(tokens)
            pred_norm = _hellaswag_helper.get_most_likely_row(tokens, mask, logits)
        num_total += 1
        # print('label', label)
        # print('pred_norm', pred_norm)
        num_correct_norm += int(pred_norm == label)
    acc_norm = num_correct_norm / num_total
    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")

if __name__ == '__main__':
    main()
