"""
dataloader.py — DataLoader with support for mixed datasets.

Single dataset usage (backward compatible):
    dl = DataLoader(NBatch=4, ContextSize=1024,
                    sources=[("data/openwebtext/EnKoMix", 1.0)],
                    split='train')

Mixed dataset usage:
    dl = DataLoader(NBatch=4, ContextSize=1024,
                    sources=[
                        ("data/openwebtext/EnKoMix", 0.7),
                        ("data/korean/EnKoMix",      0.3),
                    ],
                    split='train')

Probabilities are normalized automatically so they don't need to sum to 1.
"""

import numpy as np
import torch


class SingleDataset:
    """Wraps a single .bin file and serves sequential chunks from it."""

    def __init__(self, file_path: str):
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')
        self.current_pos = 0
        self.n_full_loops = 0

    def read_chunk(self, chunk_size: int) -> np.ndarray:
        """Return a chunk of (chunk_size + 1) tokens, wrapping if needed."""
        if self.current_pos + chunk_size + 1 > len(self.data):
            self.current_pos = 0
            self.n_full_loops += 1
        chunk = self.data[self.current_pos : self.current_pos + chunk_size + 1]
        self.current_pos += chunk_size
        return chunk


class DataLoader:

    def __init__(self, NBatch: int, ContextSize: int,
                 sources: list[tuple[str, float]], split: str):
        """
        Args:
            NBatch:      Number of sequences per batch.
            ContextSize: Token length of each sequence.
            sources:     List of (data_dir, weight) pairs.
                         Weights are normalized automatically.
            split:       'train' or 'val'.
        """
        self.NBatch      = NBatch
        self.ContextSize = ContextSize

        file_name = 'train.bin' if split == 'train' else 'val.bin'

        self.datasets: list[SingleDataset] = []
        raw_weights: list[float]           = []

        for data_dir, weight in sources:
            path = f'{data_dir}/{file_name}'
            self.datasets.append(SingleDataset(path))
            raw_weights.append(weight)

        # Normalize weights → sampling probabilities
        total = sum(raw_weights)
        self.probs = [w / total for w in raw_weights]

        print(f"DataLoader ({split}) — {len(self.datasets)} source(s):")
        for (data_dir, _), p in zip(sources, self.probs):
            n_tokens = len(self.datasets[len(self.datasets) - len(self.probs)].data) if len(self.datasets) == 1 else "?"
            print(f"  {data_dir}/{file_name}  prob={p:.3f}")

    @property
    def ChunkSize(self) -> int:
        return self.NBatch * self.ContextSize

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a dataset according to the configured probabilities,
        then return one (x, y) batch from it.
        """

        # Pick a dataset according to mixture probabilities
        idx = np.random.choice(len(self.datasets), p=self.probs)
        raw = self.datasets[idx].read_chunk(self.ChunkSize)

        tokens = torch.from_numpy(raw.astype(np.int64))
        x = tokens[:-1].view(self.NBatch, self.ContextSize)
        y = tokens[1: ].view(self.NBatch, self.ContextSize)
        return x, y

    @property
    def n_full_loops(self) -> list[int]:
        """How many times each underlying dataset has been fully consumed."""
        return [ds.n_full_loops for ds in self.datasets]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

DoTest = False
if DoTest:
    dl = DataLoader(
        NBatch=4,
        ContextSize=10,
        sources=[
            ('/Users/jskim/Documents/MLStudy/mygpt/data/openwebtext/EnKoMix', 0.6),
            # ('/path/to/openwebtext', 0.4),
        ],
        split='train',
    )
    x, y = dl.next_batch()
    print(x, y)
    x, y = dl.next_batch()
    print(x, y)
    