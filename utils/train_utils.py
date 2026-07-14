"""Training utilities with methodologically sound evaluation.

The model is selected at the epoch with the best *validation* accuracy and the
reported test metrics are computed at that epoch (no test-set peeking).
Early stopping halts training after `patience` epochs without val improvement.
"""
import copy

import numpy as np
import torch
from sklearn.metrics import f1_score

from utils.data_loader import set_seeds


@torch.no_grad()
def _masked_metrics(logits, y, mask):
    pred = logits[mask].argmax(dim=1).cpu().numpy()
    true = y[mask].cpu().numpy()
    acc = float((pred == true).mean())
    f1_macro = f1_score(true, pred, average="macro")
    return acc, f1_macro


def train_node_classifier(model, data, epochs=300, lr=0.01, weight_decay=5e-4,
                          patience=30, device="cpu", verbose=False):
    """Full-batch training with early stopping. Returns a result dict."""
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val, best_state, best_epoch = -1.0, None, -1
    bad_epochs = 0
    history = {"loss": [], "train_acc": [], "val_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        train_acc, _ = _masked_metrics(logits, data.y, data.train_mask)
        val_acc, _ = _masked_metrics(logits, data.y, data.val_mask)
        test_acc, _ = _masked_metrics(logits, data.y, data.test_mask)

        history["loss"].append(float(loss.item()))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                  f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    test_acc, test_f1 = _masked_metrics(logits, data.y, data.test_mask)

    return {
        "model": model,
        "logits": logits.cpu(),
        "val_acc": best_val,
        "test_acc": test_acc,
        "test_f1_macro": test_f1,
        "best_epoch": best_epoch,
        "history": history,
    }


def run_multi_seed(model_factory, data, seeds, epochs=300, lr=0.01,
                   weight_decay=5e-4, patience=30, device="cpu"):
    """Train `model_factory()` once per seed; return per-seed and aggregate stats."""
    accs, f1s, runs = [], [], []
    for seed in seeds:
        set_seeds(seed)
        res = train_node_classifier(model_factory(), data, epochs=epochs, lr=lr,
                                    weight_decay=weight_decay, patience=patience,
                                    device=device)
        accs.append(res["test_acc"])
        f1s.append(res["test_f1_macro"])
        runs.append(res)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "accs": accs,
        "runs": runs,
    }


def format_result(name, agg):
    return (f"{name:<28} acc {agg['acc_mean']:.4f} +/- {agg['acc_std']:.4f}   "
            f"macro-F1 {agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f}")
