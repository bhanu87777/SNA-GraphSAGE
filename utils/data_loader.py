import random

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from config import DEVICE, SEED


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_planetoid(name="Cora", root=None):
    """Load any Planetoid citation/social graph: Cora, CiteSeer or PubMed."""
    set_seeds()
    if root is None:
        root = f"data/{name.lower()}"
    dataset = Planetoid(root=root, name=name)
    data = dataset[0].to(DEVICE)
    return dataset, data


def load_cora(root="data/cora"):
    return load_planetoid("Cora", root=root)


def augment_with_structural_features(data, dataset_name="Cora"):
    """Concatenate z-scored SNA features (degree, PageRank, betweenness, ...)
    to the raw node attributes. Returns a copy; the original is untouched."""
    from utils.structural_features import compute_structural_features
    feats = compute_structural_features(
        data.edge_index, data.num_nodes,
        cache_path=f"data/structural_{dataset_name.lower()}.npy")
    feats = torch.from_numpy(feats).to(data.x.device)
    data_aug = data.clone()
    data_aug.x = torch.cat([data.x, feats], dim=1)
    return data_aug
