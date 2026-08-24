"""Sequence dataset for sleep-stage classification."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SleepSequenceDataset(Dataset):
    """Dataset of contiguous 10-epoch sequences from preprocessed caches."""

    def __init__(self, cache_index_df: pd.DataFrame, split: str,
                 seq_len: int = 10, stride: int = 5):
        self.samples = []
        self.cache = {}
        self.seq_len = seq_len

        rows = cache_index_df.loc[cache_index_df["split"] == split]
        for _, row in rows.iterrows():
            path = row["cache_path"]
            data = np.load(path)
            n = len(data["labels"])
            for start in range(0, max(n - seq_len + 1, 1), stride):
                self.samples.append((path, start))

    def _load(self, path):
        if path not in self.cache:
            d = np.load(path)
            self.cache[path] = (d["epochs"], d["labels"])
        return self.cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, start = self.samples[index]
        epochs, labels = self._load(path)

        end = min(start + self.seq_len, len(labels))
        x = epochs[start:end]
        y = labels[start:end]

        if len(x) < self.seq_len:
            pad = self.seq_len - len(x)
            x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)], axis=0)
            y = np.concatenate([y, np.repeat(y[-1:], pad, axis=0)], axis=0)

        return torch.from_numpy(x).float(), torch.from_numpy(y).long()
