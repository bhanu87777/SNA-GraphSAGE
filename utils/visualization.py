import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import FIGURES_DIR, TSNE_PERPLEXITY


def _ensure_dir(outpath):
    os.makedirs(outpath, exist_ok=True)


def _scatter_by_class(emb2, labels, title, outpath, show_legend=True):
    _ensure_dir(outpath)
    plt.figure(figsize=(8, 6))
    num_classes = int(np.max(labels)) + 1
    for c in range(num_classes):
        idx = labels == c
        plt.scatter(emb2[idx, 0], emb2[idx, 1], label=str(c), alpha=0.7, s=10)
    plt.title(title)
    if show_legend:
        plt.legend(markerscale=2)
    plt.tight_layout()
    filename = os.path.join(outpath, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    return filename


def pca_and_plot(embeddings, labels, title, outpath=FIGURES_DIR, show_legend=True):
    emb2 = PCA(n_components=2).fit_transform(embeddings)
    return _scatter_by_class(emb2, labels, title, outpath, show_legend)


def tsne_and_plot(embeddings, labels, title, outpath=FIGURES_DIR, show_legend=True, seed=42):
    if embeddings.shape[1] > 50:  # PCA pre-reduction speeds t-SNE up considerably
        embeddings = PCA(n_components=50).fit_transform(embeddings)
    emb2 = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=seed,
                init="pca").fit_transform(embeddings)
    return _scatter_by_class(emb2, labels, title, outpath, show_legend)


def plot_training_curves(history, title, outpath=FIGURES_DIR):
    _ensure_dir(outpath)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(history["loss"], color="tab:red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.set_title("Loss")
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    filename = os.path.join(outpath, f"{title.replace(' ', '_')}_curves.png")
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return filename


def plot_confusion_matrix(cm, title, class_names=None, outpath=FIGURES_DIR):
    _ensure_dir(outpath)
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(n), class_names, rotation=45)
    plt.yticks(range(n), class_names)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    filename = os.path.join(outpath, f"{title.replace(' ', '_')}_confusion.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    return filename


def plot_model_comparison(names, means, stds, title, ylabel="Test accuracy",
                          outpath=FIGURES_DIR):
    _ensure_dir(outpath)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(names))
    plt.bar(x, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.85)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    for i, m in enumerate(means):
        plt.text(i, m + 0.005, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    filename = os.path.join(outpath, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    return filename
