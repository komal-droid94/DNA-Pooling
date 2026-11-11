"""Model interpretation: turn first-layer filters into motifs and compare to the ground truth.

Method (as in DeepBind / Basset): for each filter, collect the sequence windows that activate it
strongly, stack them, and count base frequencies per position. The result is a PWM showing what
the filter "looks for".
"""

import numpy as np
import torch

from .data import BASES, motif_pwm, one_hot, reverse_complement


@torch.no_grad()
def filter_pwms(model, seqs: list[str], threshold: float = 0.5, max_seqs: int = 2000) -> list[dict]:
    """PWM for every first-layer filter, built from windows activating above threshold * max."""
    x = torch.from_numpy(np.stack([one_hot(s) for s in seqs[:max_seqs]]))
    acts = model.first_layer_activations(x).numpy()  # (n, filters, positions)
    k = model.conv1.kernel_size[0]
    results = []
    for f in range(acts.shape[1]):
        a = acts[:, f, :]
        peak = a.max()
        if peak <= 0:
            results.append({"filter": f, "pwm": None, "n_sites": 0})
            continue
        seq_idx, pos_idx = np.where(a > threshold * peak)
        if len(seq_idx) > 5000:  # keep the 5000 strongest windows
            top = np.argsort(a[seq_idx, pos_idx])[-5000:]
            seq_idx, pos_idx = seq_idx[top], pos_idx[top]
        windows = x.numpy()[seq_idx[:, None], :, pos_idx[:, None] + np.arange(k)]  # (n, k, 4)
        pwm = (windows.sum(0).T + 0.1) / (len(windows) + 0.4)  # (4, k), with pseudocounts
        results.append({"filter": f, "pwm": pwm, "n_sites": len(windows)})
    return results


def information_content(pwm: np.ndarray) -> float:
    """Total information (bits) in a PWM; a random filter scores ~0, a sharp motif ~2 bits/base."""
    return float((2 + (pwm * np.log2(pwm)).sum(axis=0)).sum())


def best_match(filter_pwm: np.ndarray, motif: str) -> tuple[float, int, str]:
    """Pearson correlation between a filter PWM and the planted motif (either strand), at the
    best offset. Returns (correlation, offset, strand)."""
    best = (-1.0, 0, "+")
    for strand, m in (("+", motif), ("-", reverse_complement(motif))):
        target = motif_pwm(m)
        k_m = target.shape[1]
        for off in range(filter_pwm.shape[1] - k_m + 1):
            window = filter_pwm[:, off:off + k_m]
            r = np.corrcoef(window.ravel(), target.ravel())[0, 1]
            if r > best[0]:
                best = (float(r), off, strand)
    return best


def consensus(pwm: np.ndarray) -> str:
    return "".join(BASES[i] for i in pwm.argmax(axis=0))
