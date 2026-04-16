"""
generate.py — Load a checkpoint and generate text from a prompt.

Usage:
    python generate.py --checkpoint logs/ckpt_01000.pt --prompt "Hello" --max_tokens 100
    python generate.py --checkpoint auto --prompt "Once upon a time" --max_tokens 200 --seed 42
"""

import os
import glob
import argparse

import torch

from model import ModelConfig, MyGPT
from utils.load_model import load_model

MODEL_BAESDIR = os.environ['MODEL_BASEDIR']
DATA_BASEDIR = os.environ['DATA_BASEDIR']
TOKENIZER_BASEDIR = os.environ['TOKENIZER_BASEDIR']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_checkpoint_path(ckpt_arg: str, log_dir: str) -> str:
    if ckpt_arg == 'auto':
        matches = sorted(glob.glob(os.path.join(log_dir, 'ckpt_*.pt')))
        if not matches:
            raise FileNotFoundError(f"No checkpoints found in '{log_dir}'")
        return matches[-1]
    if not os.path.isfile(ckpt_arg):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")
    return ckpt_arg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text from a MyGPT checkpoint")
    parser.add_argument('--checkpoint', type=str, default='auto',
                        help="Path to checkpoint file, or 'auto' for latest in --log_dir")
    parser.add_argument('--log_dir',    type=str, default='logs',
                        help="Directory to search when --checkpoint=auto (default: logs)")
    parser.add_argument('--prompt',     type=str, default=None,
                        help="Input text to condition generation on")
    parser.add_argument('--prompt_file', type=str, default=None,
                    help="Path to a .txt file containing the prompt (UTF-8)")
    parser.add_argument('--max_tokens', type=int, default=100,
                        help="Number of tokens to generate (default: 100)")
    parser.add_argument('--seed',       type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
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
    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.log_dir)
    model     = load_model(ckpt_path, device)

    # --- Prompt ---
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
        except UnicodeDecodeError:
            # Fallback for common Korean Windows encoding
            with open(args.prompt_file, 'r', encoding='cp949') as f:
                prompt_text = f.read().strip()
    else:
        prompt_text = args.prompt

    # --- Generate ---
    input_ids = encode_func(prompt_text)
    print(f"\nPrompt ({len(input_ids)} tokens):")
    print(f"  {prompt_text}")
    print(f"\nGenerating {args.max_tokens} tokens...\n")
    print("-" * 60)

    output_ids  = model.generate(input_ids, max_tokens=args.max_tokens, seed=args.seed)
    output_text = decode_func(output_ids[0].tolist())
    print(output_text)
    print("-" * 60)
    print(f"\nTotal tokens: {len(output_ids[0])} ({len(input_ids)} prompt + {args.max_tokens} generated)")


if __name__ == '__main__':
    main()
