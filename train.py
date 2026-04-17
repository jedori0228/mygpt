"""
train.py — Training script for MyGPT
All hyperparameters are configurable at the top of the file.

Usage:
    # Fresh training
    python train.py

    # Resume from a specific checkpoint
    python train.py --resume logs/ckpt_01000.pt

    # Resume from the latest checkpoint in LOG_BASEDIR automatically
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
from utils.load_model import build_optimizer, load_model

MODEL_BAESDIR = os.environ['MODEL_BASEDIR']
DATA_BASEDIR = os.environ['DATA_BASEDIR']
TOKENIZER_BASEDIR = os.environ['TOKENIZER_BASEDIR']
LOG_BASEDIR = os.environ['LOG_BASEDIR']

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tokenizer
TokenizerType = 'EnKoMix' # 'gpt2' or 'EnKoMix'

# I/O — list of (data_dir_suffix, weight) pairs; weights are normalized automatically
DATA_SOURCES = [
    (f'{DATA_BASEDIR}/openwebtext/{TokenizerType}', 0.7),
    (f'{DATA_BASEDIR}/KOREAN-WEBTEXT/{TokenizerType}', 0.3),
]

# Model
CONTEXT_SIZE    = 1024
D_HEAD          = 64
N_HEAD_Q        = 12
N_HEAD_KV       = 12
N_LAYER         = 12

# Training
N_BATCH         = 4
N_MAX_ITER      = 10000
GRAD_ACCUM_STEPS = 16

# Learning rate schedule (cosine decay with linear warmup)
LR_MAX          = 3e-4
LR_MIN          = 3e-5
WARMUP_FRAC     = 0.0033333333
WARMUP_STEPS    = int(WARMUP_FRAC*N_MAX_ITER)

# Regularization
WEIGHT_DECAY    = 0.1             # Applied to non-bias, non-norm params only
GRAD_CLIP       = 1.0             # Set to 0.0 to disable

# Logging & evaluation
LOG_EVERY       = 50
EVAL_EVERY      = 500
EVAL_ITERS      = 200
CKPT_EVERY      = 2000             # Save checkpoint every N steps

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
                    global_step: int, name: None):
    os.makedirs(LOG_BASEDIR, exist_ok=True)
    CheckPointFileName = f'ckpt_{global_step:09d}.pt' if (name is None) else f'ckpt_{name}_{global_step:09d}.pt'
    path = os.path.join(LOG_BASEDIR, CheckPointFileName)
    torch.save({
        'model':       model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'config':      config,        # raw config, not compiled model attr
        'global_step': global_step,
    }, path)
    print(f"[Checkpoint] Saved → {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # --- CLI args ---
    parser = argparse.ArgumentParser(description="Train MyGPT")
    parser.add_argument(
        '--resume', type=str, default='',
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        '--opt_lr_local',
        action='store_true',
        help="Use local step to calculate learning rate instead of Global step"
    )
    parser.add_argument(
        '--test_prompt', type=str, default='Hello, I am a language model.',
        help="Input text to test generation")
    parser.add_argument(
        '--name', type=str, default=None,
        help="Name of the job"
    )
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

    # --- Model ---

    ckpt_path = args.resume
    if ckpt_path:
        model, optimizer, GlobalStepCounter = load_model(ckpt_path, device, WEIGHT_DECAY, LR_MAX)
    else:
        model = MyGPT(config).to(device)  
        optimizer = build_optimizer(model, WEIGHT_DECAY, LR_MAX)
        GlobalStepCounter = 0

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"GlobalStepCounter: {GlobalStepCounter}")

    print(f"Compiling the model..")
    model = torch.compile(model)
    print("-> Done!")

    # --- Data ---
    print(f"Data sources: {DATA_SOURCES}")
    dataloaders = {
        split: DataLoader(N_BATCH, config.ContextSize, sources=DATA_SOURCES, split=split)
        for split in ['train', 'val']
    }

    # --- Sanity check: generate before training ---
    print("\n[Before training]")
    ids = encode_func(args.test_prompt)
    out = decode_func(model.generate(ids, max_tokens=100)[0].tolist())
    print(f"  {out}\n")

    # --- Training loop ---
    model.train()

    print("START TRAINING")
    print("- N_MAX_ITER:", N_MAX_ITER)

    for local_step in range(N_MAX_ITER):

        # GlobalStepCounter is the true position in the full training run;
        # get_lr and all schedule logic use it so LR is consistent on resume.
        last_step = (local_step == N_MAX_ITER - 1)
        lr = get_lr(local_step if args.opt_lr_local else GlobalStepCounter)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # -- Evaluation --
        if GlobalStepCounter % EVAL_EVERY == 0 or last_step:
            # Train/Val loss calculation
            losses = estimate_loss(model, dataloaders, device)
            print(f"[Evalution] step {GlobalStepCounter:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        if (GlobalStepCounter % CKPT_EVERY == 0 and local_step>0) or last_step:
            save_checkpoint(model, optimizer, config, GlobalStepCounter)

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
            print(f"[Status] step {GlobalStepCounter:5d} | loss {accumulated_loss:.6f} | lr {lr:.2e} | dt {dt_ms:.1f}ms | tok/sec {tok_per_sec:.0f}")

        GlobalStepCounter += 1

    # --- Final generation ---
    print("\n[After training]")
    ids = encode_func(args.test_prompt)
    out = decode_func(model.generate(ids, max_tokens=100)[0].tolist())
    print(f"  {out}")


if __name__ == '__main__':
    main()