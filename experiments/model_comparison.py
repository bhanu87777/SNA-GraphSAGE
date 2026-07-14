"""Compare GNN architectures (GraphSAGE, GCN, GAT, GIN) over multiple seeds."""
from config import (DEVICE, GAT_HEADS, GNN_DROPOUT, GNN_EPOCHS, GNN_HIDDEN,
                    GNN_LR, GNN_PATIENCE, GNN_WEIGHT_DECAY, SEEDS)
from models.gnn_zoo import GNN
from utils.train_utils import format_result, run_multi_seed
from utils.visualization import plot_model_comparison

ARCHS = ["sage", "gcn", "gat", "gin"]
ARCH_LABELS = {"sage": "GraphSAGE", "gcn": "GCN", "gat": "GAT", "gin": "GIN"}


def model_comparison_experiment(dataset, data, seeds=SEEDS, epochs=GNN_EPOCHS):
    in_dim, out_dim = dataset.num_node_features, dataset.num_classes
    results = {}
    for arch in ARCHS:
        def factory(a=arch):
            return GNN(a, in_dim, GNN_HIDDEN, out_dim, num_layers=2,
                       dropout=GNN_DROPOUT, heads=GAT_HEADS)
        agg = run_multi_seed(factory, data, seeds, epochs=epochs, lr=GNN_LR,
                             weight_decay=GNN_WEIGHT_DECAY, patience=GNN_PATIENCE,
                             device=DEVICE)
        results[arch] = agg
        print(format_result(ARCH_LABELS[arch], agg))

    names = [ARCH_LABELS[a] for a in ARCHS]
    means = [results[a]["acc_mean"] for a in ARCHS]
    stds = [results[a]["acc_std"] for a in ARCHS]
    fig = plot_model_comparison(names, means, stds, "GNN_architecture_comparison")
    print("Saved comparison chart to:", fig)
    return results


def aggregator_experiment(dataset, data, seeds=SEEDS, epochs=GNN_EPOCHS):
    """The paper compares mean/pool/LSTM aggregators; we compare the
    aggregators available in PyG's SAGEConv plus GAT-style attention."""
    in_dim, out_dim = dataset.num_node_features, dataset.num_classes
    configs = [
        ("SAGE-mean", dict(arch="sage", aggr="mean")),
        ("SAGE-max (pool)", dict(arch="sage", aggr="max")),
        ("SAGE-sum", dict(arch="sage", aggr="sum")),
        ("GAT (attention)", dict(arch="gat", aggr="mean")),
    ]
    results = {}
    for name, cfg in configs:
        def factory(c=cfg):
            return GNN(c["arch"], in_dim, GNN_HIDDEN, out_dim, num_layers=2,
                       dropout=GNN_DROPOUT, aggr=c["aggr"], heads=GAT_HEADS)
        agg = run_multi_seed(factory, data, seeds, epochs=epochs, lr=GNN_LR,
                             weight_decay=GNN_WEIGHT_DECAY, patience=GNN_PATIENCE,
                             device=DEVICE)
        results[name] = agg
        print(format_result(name, agg))

    names = list(results.keys())
    means = [results[n]["acc_mean"] for n in names]
    stds = [results[n]["acc_std"] for n in names]
    fig = plot_model_comparison(names, means, stds, "Aggregator_comparison")
    print("Saved aggregator chart to:", fig)
    return results
