"""Training loop with early stopping, plus evaluation metrics (accuracy, ROC-AUC)."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney U) formula, with ties averaged."""
    y_true = np.asarray(y_true).astype(bool)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):  # assign average rank to ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def predict(model: nn.Module, ds: TensorDataset, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, y in DataLoader(ds, batch_size=batch_size):
        probs.append(torch.sigmoid(model(x)).numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def evaluate(model: nn.Module, ds: TensorDataset) -> dict:
    probs, y = predict(model, ds)
    return {"accuracy": float(((probs > 0.5) == y).mean()), "auc": roc_auc(y, probs)}


def train(model: nn.Module, train_ds: TensorDataset, val_ds: TensorDataset, epochs: int = 25,
          lr: float = 3e-3, batch_size: int = 128, patience: int = 4, seed: int = 0,
          verbose: bool = True) -> dict:
    """Adam + binary cross-entropy. Keeps the weights from the best validation-loss epoch."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_loss, best_state, bad_epochs, history = float("inf"), None, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / len(train_ds)

        model.eval()
        with torch.no_grad():
            xv, yv = val_ds.tensors
            val_loss = loss_fn(model(xv), yv).item()
        metrics = evaluate(model, val_ds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        if verbose:
            print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                  f"val_acc {metrics['accuracy']:.4f}  val_auc {metrics['auc']:.4f}")

        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_loss}
