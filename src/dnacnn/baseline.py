"""Oracle baseline: scan each sequence with the TRUE motif's PWM (log-odds vs background).

This classifier is told the answer (the planted motif), so it is close to the best any model can
do on the synthetic task. Comparing the CNN to it shows how much of the achievable performance
the CNN reached while learning the motif from scratch.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .data import motif_pwm, one_hot, reverse_complement

BACKGROUND = np.array([0.29, 0.21, 0.21, 0.29])[:, None]  # A, C, G, T at 42% GC


def pwm_scan_scores(seqs: list[str], motif: str, strength: float) -> np.ndarray:
    """Best log-odds match score per sequence, over all positions on both strands."""
    log_odds = np.log(motif_pwm(motif, strength) / BACKGROUND)  # (4, k)
    k = log_odds.shape[1]
    scores = np.empty(len(seqs))
    for i, seq in enumerate(seqs):
        best = -np.inf
        for s in (seq, reverse_complement(seq)):
            windows = sliding_window_view(one_hot(s), k, axis=1)  # (4, L-k+1, k)
            best = max(best, (windows * log_odds[:, None, :]).sum(axis=(0, 2)).max())
        scores[i] = best
    return scores
