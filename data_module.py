"""
Sequence-window Dataset for RegTrust-XAI: reads one-hot DNA sequence on
demand from the 2bit genome at train/eval time, joined against the
coordinate/label table build_features.py already produced.

WHY LAZY, PER-WORKER GENOME READS. phase1_sequence_windows.npz stores only
coordinates and labels (chrom/start/end/label/is_positive), not
materialised one-hot arrays -- see build_features.py's module docstring for
why (several GB for ~500k windows, no benefit). The same "cache/read in
worker" discipline ENZYME_XAI's data_module.py uses for AlphaFold structures
applies here to genome coordinates: each DataLoader worker process opens its
own py2bit handle lazily (worker_init_fn below), because a py2bit handle is
a C-extension file handle that is not safe to share across a fork the way a
plain Python object would be -- opening it in __init__ before the
DataLoader forks workers would hand every worker the SAME underlying file
descriptor, which is exactly the kind of subtle multi-process bug that only
shows up as corrupted reads under load, not an import error.
"""
from __future__ import annotations

import numpy as np
import py2bit
import torch
from torch.utils.data import Dataset

from config import ALPHABET, GENOME_2BIT, N_CHANNELS, WINDOW_BP

BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}


def one_hot_encode(seq: str) -> np.ndarray:
    """(N_CHANNELS, len(seq)) float32 one-hot. Ambiguous/N bases (real,
    non-negligible in a genome -- centromeric and telomeric regions in
    particular) get the all-zero column, not a uniform 0.25 -- an all-zero
    encoding is what ChromBPNet-lineage tools use by convention, and it lets
    the network's own bias terms represent "no information here" distinctly
    from "A/C/G/T observed", which a uniform 0.25 column would not."""
    seq = seq.upper()
    arr = np.zeros((N_CHANNELS, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        idx = BASE_TO_IDX.get(b)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


class WindowDataset(Dataset):
    """indices: which rows of the windows npz this instance serves (a fold's
    train or val split, or the held-out chromosome test set) -- splitting is
    the CALLER's responsibility (splits.py), this class only reads.

    label_mean/label_std: fit per-fold on that fold's OWN training indices
    only (see train_cv.py) -- this dataset never re-derives its own
    standardization, so passing the wrong stats is a caller error, not
    something this class can silently get right on its own. Default to
    mean=0/std=1 (i.e. raw, unstandardized labels) for callers that only
    need the true label value back, not a z-scored training target -- eval
    code de-standardizes each model's own prediction with that model's own
    fold-local scaler instead, so the loader itself has no scaler to apply.
    """

    def __init__(self, windows_npz, indices, label_mean: float = 0.0, label_std: float = 1.0,
                 genome_2bit_path=GENOME_2BIT, window_bp=WINDOW_BP):
        self.chrom = windows_npz["chrom"][indices]
        self.start = windows_npz["start"][indices]
        self.end = windows_npz["end"][indices]
        self.label = windows_npz["label"][indices].astype(np.float32)
        self.label_mean = float(label_mean)
        self.label_std = float(label_std) if label_std > 1e-8 else 1.0
        self.genome_2bit_path = str(genome_2bit_path)
        self.window_bp = int(window_bp)
        self._tb = None  # opened lazily, per-process -- see worker_init_fn

    def _genome(self):
        if self._tb is None:
            self._tb = py2bit.open(self.genome_2bit_path)
        return self._tb

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        tb = self._genome()
        seq = tb.sequence(str(self.chrom[i]), int(self.start[i]), int(self.end[i]))
        if len(seq) != self.window_bp:
            # Should not happen -- build_features.py only kept windows that fit
            # inside their chromosome -- but fail loudly rather than silently
            # feed the model a wrong-shaped tensor if it ever does.
            raise ValueError(
                f"window {self.chrom[i]}:{self.start[i]}-{self.end[i]} returned "
                f"{len(seq)}bp, expected {self.window_bp}bp"
            )
        x = torch.from_numpy(one_hot_encode(seq))
        y = (self.label[i] - self.label_mean) / self.label_std
        return x, torch.tensor(y, dtype=torch.float32)


def worker_init_fn(_worker_id):
    """Force each DataLoader worker to open its own py2bit handle on first
    use rather than inheriting one across the fork -- see module docstring.
    Nothing to do here explicitly; leaving self._tb as None at fork time and
    lazily opening in _genome() already gives each worker process its own
    handle the first time __getitem__ runs in that process. This function
    exists so that guarantee is documented and wired in explicitly (passed
    to DataLoader(worker_init_fn=...)) rather than relying on an implicit
    default that a future edit could break silently."""
    return None
