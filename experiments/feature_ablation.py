"""Ablation: raw attributes vs. SNA structural features vs. both combined.

Tests whether classic social-network-analysis descriptors (PageRank,
betweenness, clustering coefficient, k-core, ...) carry signal that the raw
node attributes do not.
"""
import torch

from config import (DEVICE, GNN_DROPOUT, GNN_EPOCHS, GNN_HIDDEN, GNN_LR,
                    GNN_PATIENCE, GNN_WEIGHT_DECAY, SEEDS)
from models.gnn_zoo import GNN
from utils.data_loader import augment_with_structural_features
from utils.structural_features import compute_structural_features
from utils.train_utils import format_result, run_multi_seed
from utils.visualization import plot_model_comparison


def feature_ablation_experiment(dataset, data, dataset_name="Cora",
                                seeds=SEEDS, epochs=GNN_EPOCHS):
    out_dim = dataset.num_classes

    data_aug = augment_with_structural_features(data, dataset_name)

    struct = compute_structural_features(
        data.edge_index, data.num_nodes,
        cache_path=f"data/structural_{dataset_name.lower()}.npy")
    data_struct = data.clone()
    data_struct.x = torch.from_numpy(struct).to(data.x.device)

    variants = [
        ("Raw attributes", data),
        ("SNA features only", data_struct),
        ("Raw + SNA features", data_aug),
    ]
    results = {}
    for name, d in variants:
        in_dim = d.num_features
        def factory(dim=in_dim):
            return GNN("sage", dim, GNN_HIDDEN, out_dim, num_layers=2,
                       dropout=GNN_DROPOUT)
        agg = run_multi_seed(factory, d, seeds, epochs=epochs, lr=GNN_LR,
                             weight_decay=GNN_WEIGHT_DECAY, patience=GNN_PATIENCE,
                             device=DEVICE)
        results[name] = agg
        print(format_result(name, agg))

    names = list(results.keys())
    means = [results[n]["acc_mean"] for n in names]
    stds = [results[n]["acc_std"] for n in names]
    fig = plot_model_comparison(names, means, stds, "SNA_feature_ablation")
    print("Saved ablation chart to:", fig)
    return results
