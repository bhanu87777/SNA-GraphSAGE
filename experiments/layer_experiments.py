"""Depth experiment: how many message-passing layers before oversmoothing?"""
from config import (DEVICE, GNN_DROPOUT, GNN_EPOCHS, GNN_HIDDEN, GNN_LR,
                    GNN_PATIENCE, GNN_WEIGHT_DECAY, SEEDS)
from models.gnn_zoo import GNN
from utils.train_utils import format_result, run_multi_seed
from utils.visualization import plot_model_comparison


def layer_experiment(dataset, data, layer_list=(1, 2, 3, 4), seeds=SEEDS,
                     epochs=GNN_EPOCHS):
    in_dim, out_dim = dataset.num_node_features, dataset.num_classes
    results = {}
    for L in layer_list:
        def factory(layers=L):
            return GNN("sage", in_dim, GNN_HIDDEN, out_dim,
                       num_layers=layers, dropout=GNN_DROPOUT)
        agg = run_multi_seed(factory, data, seeds, epochs=epochs, lr=GNN_LR,
                             weight_decay=GNN_WEIGHT_DECAY, patience=GNN_PATIENCE,
                             device=DEVICE)
        results[L] = agg
        print(format_result(f"GraphSAGE {L} layer(s)", agg))

    names = [f"{L} layer(s)" for L in layer_list]
    means = [results[L]["acc_mean"] for L in layer_list]
    stds = [results[L]["acc_std"] for L in layer_list]
    fig = plot_model_comparison(names, means, stds, "Depth_vs_accuracy")
    print("Saved depth chart to:", fig)
    return results
