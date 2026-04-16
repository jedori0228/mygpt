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
import tiktoken

from model import ModelConfig, MyGPT


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


def load_model(ckpt_path: str, device: str) -> MyGPT:
    # PyTorch 2.6+ defaults weights_only=True, which blocks custom classes.
    # Allowlisting ModelConfig is the safe fix — avoids arbitrary code execution
    # while still supporting our known dataclass.
    torch.serialization.add_safe_globals([ModelConfig])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    config = ckpt['config']
    model  = MyGPT(config).to(device)
    # Strip _orig_mod. prefix if checkpoint was saved from a torch.compile()d model
    state_dict = ckpt["model"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    step     = ckpt.get('global_step', '?')
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  global_step={step} | val_loss={val_loss:.4f}")
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text from a MyGPT checkpoint")
    parser.add_argument('--checkpoint', type=str, default='auto',
                        help="Path to checkpoint file, or 'auto' for latest in --log_dir")
    parser.add_argument('--log_dir',    type=str, default='logs',
                        help="Directory to search when --checkpoint=auto (default: logs)")
    parser.add_argument('--prompt',     type=str, required=True,
                        help="Input text to condition generation on")
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
    enc    = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s)
    decode = lambda ids: enc.decode(ids)

    # --- Load model ---
    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.log_dir)
    model     = load_model(ckpt_path, device)

    # --- Generate ---
    input_ids = encode(args.prompt)
    print(f"\nPrompt ({len(input_ids)} tokens):")
    print(f"  {args.prompt}")
    print(f"\nGenerating {args.max_tokens} tokens...\n")
    print("-" * 60)

    output_ids  = model.generate(input_ids, max_tokens=args.max_tokens, seed=args.seed)
    output_text = decode(output_ids[0].tolist())
    print(output_text)
    print("-" * 60)
    print(f"\nTotal tokens: {len(output_ids[0])} ({len(input_ids)} prompt + {args.max_tokens} generated)")


if __name__ == '__main__':
    main()
