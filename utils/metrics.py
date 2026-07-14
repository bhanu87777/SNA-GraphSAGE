import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)


def evaluate_logits(logits, labels, mask):
    preds = logits.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = mask.cpu().numpy()
    acc = accuracy_score(labels[mask], preds[mask])
    report = classification_report(labels[mask], preds[mask])
    return acc, report


def full_metrics(logits, labels, mask):
    """Accuracy, micro/macro F1 and confusion matrix on the masked nodes."""
    preds = logits.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = mask.cpu().numpy()
    y_true, y_pred = labels[mask], preds[mask]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred),
    }
