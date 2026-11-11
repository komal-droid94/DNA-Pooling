import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dnacnn import data, interpret  # noqa: E402
from dnacnn.baseline import pwm_scan_scores  # noqa: E402
from dnacnn.model import DNACNN  # noqa: E402
from dnacnn.train import roc_auc  # noqa: E402


def test_one_hot_encoding():
    x = data.one_hot("ACGTN")
    assert x.shape == (4, 5)
    assert x[:, :4].tolist() == np.eye(4).tolist()
    assert x[:, 4].sum() == 0  # N -> all zeros


def test_reverse_complement():
    assert data.reverse_complement("AGATAAGG") == "CCTTATCT"
    assert data.reverse_complement(data.reverse_complement("ACGTTG")) == "ACGTTG"


def test_read_fasta(tmp_path):
    f = tmp_path / "x.fa"
    f.write_text(">seq1\nACGT\nAC\n>seq2\nTTTT\n")
    assert data.read_fasta(f) == ["ACGTAC", "TTTT"]


def test_synthetic_dataset_balanced_and_planted():
    seqs, labels, pos = data.make_synthetic_dataset(n=200, strength=1.0, seed=1)
    assert labels.sum() == 100 and len(set(map(len, seqs))) == 1
    motif, rc = data.DEFAULT_MOTIF, data.reverse_complement(data.DEFAULT_MOTIF)
    for s, y, p in zip(seqs, labels, pos):
        if y == 1:
            assert s[p:p + len(motif)] in (motif, rc)


def test_position_range_respected():
    _, labels, pos = data.make_synthetic_dataset(n=200, position_range=(80, 120), seed=2)
    assert pos[labels == 1].min() >= 80 and pos[labels == 1].max() <= 120


@pytest.mark.parametrize("pooling", ["max", "avg", "none"])
def test_model_output_shape(pooling):
    model = DNACNN(seq_len=200, pooling=pooling)
    out = model(torch.zeros(3, 4, 200))
    assert out.shape == (3,)


def test_global_max_pooling_is_shift_invariant():
    """Moving the motif should barely change the max-pooled model's output (no training needed)."""
    torch.manual_seed(0)
    model = DNACNN(seq_len=200, pooling="max").eval()
    bg = "A" * 200
    a = bg[:30] + data.DEFAULT_MOTIF + bg[38:]
    b = bg[:150] + data.DEFAULT_MOTIF + bg[158:]
    xa, xb = (torch.from_numpy(data.one_hot(s))[None] for s in (a, b))
    with torch.no_grad():
        assert torch.allclose(model(xa), model(xb), atol=1e-4)


def test_roc_auc_matches_known_values():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75


def test_oracle_baseline_separates_classes():
    seqs, labels, _ = data.make_synthetic_dataset(n=400, seed=3)
    assert roc_auc(labels, pwm_scan_scores(seqs, data.DEFAULT_MOTIF, 0.9)) > 0.8


def test_best_match_finds_exact_motif():
    pwm = data.motif_pwm("TT" + data.DEFAULT_MOTIF + "TT", 0.9)
    r, offset, strand = interpret.best_match(pwm, data.DEFAULT_MOTIF)
    assert r > 0.99 and offset == 2 and strand == "+"
