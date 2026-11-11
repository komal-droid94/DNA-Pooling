"""DNA data utilities: one-hot encoding, FASTA reading, and a synthetic motif dataset.

The synthetic task mirrors real transcription-factor (TF) binding prediction (e.g. DeepBind,
Alipanahi et al. 2015): a sequence is "bound" (label 1) if it contains a binding motif somewhere,
and the position of the motif varies. Because we plant the motif ourselves, we know the ground
truth and can check whether the CNN's filters actually learned it.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

BASES = "ACGT"
BASE_TO_INDEX = {b: i for i, b in enumerate(BASES)}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

# GATA-family binding site consensus (WGATAR). Used as the planted motif.
DEFAULT_MOTIF = "AGATAAGG"
DEFAULT_STRENGTH = 0.9  # probability each motif position shows the consensus base


def one_hot(seq: str) -> np.ndarray:
    """Encode a DNA string as a (4, L) matrix. Unknown bases (N) become all zeros."""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        idx = BASE_TO_INDEX.get(base)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(COMPLEMENT)[::-1]


def read_fasta(path: str | Path) -> list[str]:
    """Minimal FASTA reader: returns the sequences, ignoring header lines."""
    seqs, current = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current:
                seqs.append("".join(current))
            current = []
        elif line:
            current.append(line)
    if current:
        seqs.append("".join(current))
    return seqs


def motif_pwm(consensus: str, strength: float = 0.9) -> np.ndarray:
    """Position weight matrix (4, k): each position emits the consensus base with p=strength."""
    pwm = np.full((4, len(consensus)), (1 - strength) / 3, dtype=np.float64)
    for j, base in enumerate(consensus):
        pwm[BASE_TO_INDEX[base], j] = strength
    return pwm


def sample_from_pwm(pwm: np.ndarray, rng: np.random.Generator) -> str:
    return "".join(BASES[rng.choice(4, p=pwm[:, j])] for j in range(pwm.shape[1]))


def random_sequence(length: int, rng: np.random.Generator, gc: float = 0.42) -> str:
    """Background DNA with roughly human-genome GC content (~42%)."""
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A, C, G, T
    return "".join(rng.choice(list(BASES), size=length, p=p))


def make_synthetic_dataset(n: int = 10_000, length: int = 200, motif: str = DEFAULT_MOTIF,
                           strength: float = 0.9, position_range: tuple[int, int] | None = None,
                           seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Half the sequences contain one noisy copy of the motif (label 1), half do not (label 0).

    position_range limits where the motif can be planted, e.g. (80, 120) keeps it near the center.
    The motif is inserted on either strand, like real binding sites.
    Returns (sequences, labels, motif_start_positions; -1 for negatives).
    """
    rng = np.random.default_rng(seed)
    pwm = motif_pwm(motif, strength)
    k = len(motif)
    lo, hi = position_range or (0, length - k)
    seqs, labels, positions = [], [], []
    for i in range(n):
        seq = random_sequence(length, rng)
        if i % 2 == 0:
            site = sample_from_pwm(pwm, rng)
            if rng.random() < 0.5:
                site = reverse_complement(site)
            pos = int(rng.integers(lo, hi + 1))
            seq = seq[:pos] + site + seq[pos + k:]
            labels.append(1)
            positions.append(pos)
        else:
            labels.append(0)
            positions.append(-1)
        seqs.append(seq)
    order = rng.permutation(n)
    return ([seqs[i] for i in order], np.array(labels)[order], np.array(positions)[order])


def to_tensor_dataset(seqs: list[str], labels: np.ndarray) -> TensorDataset:
    x = torch.from_numpy(np.stack([one_hot(s) for s in seqs]))
    y = torch.from_numpy(labels.astype(np.float32))
    return TensorDataset(x, y)
