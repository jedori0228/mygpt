"""
train.py — Training script for MyGPT
All hyperparameters are configurable at the top of the file.

Usage:
    # Fresh training
    python train.py

    # Resume from a specific checkpoint
    python train.py --resume logs/ckpt_01000.pt

    # Resume from the latest checkpoint in LOG_DIR automatically
    python train.py --resume auto
"""

import os
import math
import time
import argparse
import glob

import torch
from torch import nn
from torch.nn import functional as F
import tiktoken

from model import ModelConfig, MyGPT
from dataloader import DataLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# I/O
DATA_TYPE       = 'shakespeare'   # 'shakespeare' or 'openwebtext'
LOG_DIR         = 'logs'

# Tokenizer
TokenizerType = 'gpt2' # 'gpt2' or 'EnKoMix'

# Model
CONTEXT_SIZE    = 1024
D_HEAD          = 64
N_HEAD_Q        = 12
N_HEAD_KV       = 12               # Set < N_HEAD_Q to enable GQA
N_LAYER         = 12

# Training
N_BATCH         = 4
N_MAX_ITER      = 100
GRAD_ACCUM_STEPS = 16              # Effective batch = N_BATCH * GRAD_ACCUM_STEPS

# Learning rate schedule (cosine decay with linear warmup)
LR_MAX          = 3e-4
LR_MIN          = 3e-5
WARMUP_STEPS    = 100

# Regularization
WEIGHT_DECAY    = 0.1             # Applied to non-bias, non-norm params only
GRAD_CLIP       = 1.0             # Set to 0.0 to disable

# Logging & evaluation
LOG_EVERY       = 50
EVAL_EVERY      = 500
EVAL_ITERS      = 100
CKPT_EVERY      = 500             # Save checkpoint every N steps

# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR_MAX * (step + 1) / WARMUP_STEPS
    if step >= N_MAX_ITER:
        return LR_MIN
    progress = (step - WARMUP_STEPS) / (N_MAX_ITER - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Optimizer: separate param groups to skip weight decay on norm/bias params
# ---------------------------------------------------------------------------

def build_optimizer(model: MyGPT) -> torch.optim.AdamW:
    decay_params     = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() < 2]
    param_groups = [
        {"params": decay_params,    "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    n_decay   = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {n_decay:,} params with decay, {n_nodecay:,} without.")
    return torch.optim.AdamW(param_groups, lr=LR_MAX, betas=(0.9, 0.95), eps=1e-8)


# ---------------------------------------------------------------------------
# Loss estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, dataloaders, device) -> dict:
    model.eval()
    out = {}
    for split, dl in dataloaders.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = dl.next_batch()
            X, Y = X.to(device), Y.to(device)
            loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: MyGPT, optimizer, config: ModelConfig,
                    global_step: int, val_loss: float):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f'ckpt_{global_step:05d}.pt')
    torch.save({
        'model':       model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'config':      config,        # raw config, not compiled model attr
        'global_step': global_step,
        'val_loss':    val_loss,
    }, path)
    print(f"  Checkpoint saved → {path}")


def resolve_checkpoint_path(resume_arg: str) -> str | None:
    """
    Resolve --resume argument to an actual file path.
    - 'auto'        : pick the highest-numbered ckpt_*.pt in LOG_DIR
    - a file path   : use it directly
    - None / ''     : fresh training
    """
    if not resume_arg:
        return None
    if resume_arg == 'auto':
        matches = sorted(glob.glob(os.path.join(LOG_DIR, 'ckpt_*.pt')))
        if not matches:
            raise FileNotFoundError(f"No checkpoints found in '{LOG_DIR}' for --resume auto")
        return matches[-1]
    if not os.path.isfile(resume_arg):
        raise FileNotFoundError(f"Checkpoint not found: {resume_arg}")
    return resume_arg


def load_checkpoint(path: str, model: MyGPT, optimizer, device) -> tuple[int, float]:
    """
    Load model + optimizer state from a checkpoint.
    Returns (global_step, best_val_loss) so the training loop can resume correctly.
    """
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    # Load into the underlying model if torch.compile wraps it
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    global_step   = ckpt['global_step']
    best_val_loss = ckpt.get('val_loss', float('inf'))
    print(f"  Resumed at global_step={global_step}, val_loss={best_val_loss:.4f}")
    return global_step, best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    MODEL_BAESDIR = os.environ['MODEL_BASEDIR']
    DATA_BASEDIR = os.envicron['DATA_BASEDIR']
    TOKENIZER_BASEDIR = os.envicron['TOKENIZER_BASEDIR']

    # --- CLI args ---
    parser = argparse.ArgumentParser(description="Train MyGPT")
    parser.add_argument(
        '--resume', type=str, default='',
        help="Path to checkpoint to resume from, or 'auto' for the latest in LOG_DIR."
    )
    parser.add_argument(
        '--test_prompt', type=str, default='Hello, I am a language model.',
        help="Input text to test generation")
    args = parser.parse_args()

    # --- Device ---
    device = (torch.accelerator.current_accelerator().type
              if torch.accelerator.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Tokenizer ---
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

    # --- Model config ---
    config              = ModelConfig()
    config.DToken       = NVocab
    config.ContextSize  = CONTEXT_SIZE
    config.DHead        = D_HEAD
    config.NHeadQ       = N_HEAD_Q
    config.NHeadKV      = N_HEAD_KV
    config.NLayer       = N_LAYER

    torch.set_float32_matmul_precision('high')

    # --- Model + optimizer (must be built BEFORE loading checkpoint) ---
    model = MyGPT(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model     = torch.compile(model)
    optimizer = build_optimizer(model)

    # --- Resume (if requested) ---
    GlobalStepCounter = 0
    best_val_loss     = float('inf')

    ckpt_path = resolve_checkpoint_path(args.resume)
    if ckpt_path:
        GlobalStepCounter, best_val_loss = load_checkpoint(
            ckpt_path, model, optimizer, device
        )

    # Remaining iterations to run from this point
    remaining_steps = N_MAX_ITER - GlobalStepCounter
    if remaining_steps <= 0:
        print(f"GlobalStepCounter ({GlobalStepCounter}) >= N_MAX_ITER ({N_MAX_ITER}). Nothing to do.")
        return

    # --- Data ---
    data_dir    = DATA_BASEDIR
    dataloaders = {
        split: DataLoader(N_BATCH, config.ContextSize, data_dir, split)
        for split in ['train', 'val']
    }

    # --- Sanity check: generate before training ---
    print("\n[Before training]")
    ids = encode_func(args.test_prompt)
    out = decode_func(model.generate(ids, max_tokens=100)[0].tolist())
    print(f"  {out}\n")

    # --- Training loop ---
    model.train()

    for local_step in range(remaining_steps):

        # GlobalStepCounter is the true position in the full training run;
        # get_lr and all schedule logic use it so LR is consistent on resume.
        last_step = (local_step == remaining_steps - 1)
        lr = get_lr(GlobalStepCounter)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # -- Evaluation --
        if GlobalStepCounter % EVAL_EVERY == 0 or last_step:
            losses = estimate_loss(model, dataloaders, device)
            print(f"step {GlobalStepCounter:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(model, optimizer, config, GlobalStepCounter, best_val_loss)
            elif GlobalStepCounter % CKPT_EVERY == 0 and GlobalStepCounter > 0:
                save_checkpoint(model, optimizer, config, GlobalStepCounter, losses['val'])

        # -- Forward + backward with gradient accumulation --
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        t0 = time.time()
        for micro_step in range(GRAD_ACCUM_STEPS):
            xb, yb = dataloaders['train'].next_batch()
            xb, yb = xb.to(device), yb.to(device)
            loss   = model(xb, yb) / GRAD_ACCUM_STEPS
            loss.backward()
            accumulated_loss += loss.item()

        if GRAD_CLIP > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        if device == 'mps':
            torch.mps.synchronize()
        t1 = time.time()

        if GlobalStepCounter % LOG_EVERY == 0 or last_step:
            dt_ms        = (t1 - t0) * 1000
            tokens_total = N_BATCH * config.ContextSize * GRAD_ACCUM_STEPS
            tok_per_sec  = tokens_total / (t1 - t0)
            print(f"  step {GlobalStepCounter:5d} | loss {accumulated_loss:.6f} | lr {lr:.2e} | dt {dt_ms:.1f}ms | tok/sec {tok_per_sec:.0f}")

        GlobalStepCounter += 1

    # --- Final generation ---
    print("\n[After training]")
    ids = encode_func(args.test_prompt)
    out = decode_func(model.generate(ids, max_tokens=100)[0].tolist())
    print(f"  {out}")
    print(f"\nBest val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
