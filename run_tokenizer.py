"""
run_tokenizer.py — Train a bilingual (EN+KO) BPE tokenizer.

The mixture ratio is configurable. Each language stream is consumed
independently so one running dry does not silently truncate the other.
"""

import os
import itertools
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOCAB_SIZE   = 50257
NUM_SAMPLES  = 200000   # total documents across all languages

# Mixture: must sum to 1.0
LANG_MIX = {
    "en": 0.5,
    "ko": 0.5,
}

SAVE_DIR = "tokenizer/EnKoMix"


SPECIAL_TOKENS = [
    "<s>", "<pad>", "</s>", "<unk>", "<mask>",
    "<|jaesung|>", "<|sihyun|>",   # speaker tokens
    "<|user|>", "<|assistant|>",   # optional: generic roles for future use
]
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


# ---------------------------------------------------------------------------
# Dataset iterator with explicit per-language quota
# ---------------------------------------------------------------------------

def build_iterator(num_samples: int, lang_mix: dict):
    """
    Yields text documents sampled from each language according to lang_mix.
    Each language stream is consumed independently — one running dry does
    not stop the others.
    """
    quotas = {lang: int(num_samples * frac) for lang, frac in lang_mix.items()}
    print(f"Per-language quotas: {quotas}")

    streams = {}
    if "en" in quotas:
        streams["en"] = (
            doc["text"]
            for doc in load_dataset("openwebtext", split="train", streaming=True)
        )
    if "ko" in quotas:
        streams["ko"] = (
            doc["text"]
            for doc in load_dataset("HAERAE-HUB/KOREAN-WEBTEXT", split="train", streaming=True)
        )

    # Interleave by yielding from each stream up to its quota
    iters = {
        lang: itertools.islice(stream, quotas[lang])
        for lang, stream in streams.items()
    }
    # Round-robin across languages so the vocab sees both early in training
    active = list(iters.values())
    exhausted = set()
    i = 0
    total = 0
    while len(exhausted) < len(active):
        it = active[i % len(active)]
        if i % len(active) not in exhausted:
            try:
                text = next(it)
                yield text
                total += 1
            except StopIteration:
                exhausted.add(i % len(active))
        i += 1

    print(f"Iterator finished. Total documents yielded: {total}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    print(f"Training tokenizer — vocab_size={VOCAB_SIZE}, num_samples={NUM_SAMPLES}")
    print(f"Language mix: {LANG_MIX}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        build_iterator(NUM_SAMPLES, LANG_MIX),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
    )

    # Add BOS/EOS post-processor so encode() automatically wraps sequences.
    # This means every encoded sequence starts with <s> and ends with </s>,
    # which is important for the language model to learn sentence boundaries.
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0",
        special_tokens=[
            (BOS_TOKEN, bos_id),
            (EOS_TOKEN, eos_id),
        ],
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Save as a single self-contained tokenizer.json
    save_path = os.path.join(SAVE_DIR, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"Saved tokenizer → {save_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()