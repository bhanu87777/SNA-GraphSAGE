"""Mini-batch neighbor sampling: fixed size vs. the paper's degree-adaptive rule.

The paper (Sec. IV-C) proposes sampling S_i = c + [N_i * w] neighbors of node i
(N_i = its degree), capped at a maximum, so hub nodes contribute proportionally
more neighbors. Their Weibo graph has hubs with millions of followers so
w=0.001 works there; citation graphs have small degrees, so w is a config knob
(ADAPTIVE_W) scaled to the dataset.
"""
import random

import torch

from config import (ADAPTIVE_C, ADAPTIVE_CAP, ADAPTIVE_W, DEVICE,
                    GRAPH_SAGE_WEIGHT_DECAY)
from models.gnn_zoo import GNN
from utils.data_loader import set_seeds


def build_adj_list(edge_index, num_nodes):
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[s].append(d)
    return adj


def adaptive_sample_size(degree, c=ADAPTIVE_C, w=ADAPTIVE_W, cap=ADAPTIVE_CAP):
    """S_i = clip(c + [N_i * w], 1, cap) - the paper's variable-sampling rule."""
    return max(1, min(cap, c + int(degree * w)))


def sample_subgraph(adj, seed_nodes, sizes, adaptive=False):
    """BFS-style layered sampling. `sizes` gives the per-hop fixed fan-out;
    with adaptive=True the fan-out of each node is degree-dependent instead."""
    frontier = list(seed_nodes)
    all_nodes = set(seed_nodes)
    for size in sizes:
        new_frontier = []
        for u in frontier:
            neighs = adj[u]
            if len(neighs) == 0:
                continue
            k = adaptive_sample_size(len(neighs)) if adaptive else size
            sampled = neighs if len(neighs) <= k else random.sample(neighs, k)
            for v in sampled:
                if v not in all_nodes:
                    new_frontier.append(v)
                    all_nodes.add(v)
        frontier = new_frontier
    nodes_sorted = sorted(all_nodes)
    node_id_map = {old: i for i, old in enumerate(nodes_sorted)}
    rows, cols = [], []
    for old_u in nodes_sorted:
        for old_v in adj[old_u]:
            if old_v in node_id_map:
                rows.append(node_id_map[old_u])
                cols.append(node_id_map[old_v])
    edge_index_sub = (torch.tensor([rows, cols], dtype=torch.long)
                      if rows else torch.empty((2, 0), dtype=torch.long))
    seed_pos = [node_id_map[s] for s in seed_nodes if s in node_id_map]
    seed_mask = torch.zeros(len(nodes_sorted), dtype=torch.bool)
    seed_mask[seed_pos] = True
    return nodes_sorted, node_id_map, edge_index_sub, seed_mask


def train_with_sampling(data, num_layers=2, sample_sizes=(10, 10), adaptive=False,
                        epochs=200, hidden=128, lr=0.01, seed=42):
    set_seeds(seed)
    num_nodes = data.num_nodes
    adj = build_adj_list(data.edge_index, num_nodes)
    train_nodes = [int(i) for i in torch.where(data.train_mask.cpu())[0].tolist()]
    model = GNN("sage", data.num_features, hidden,
                int(data.y.max().item()) + 1, num_layers=num_layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=GRAPH_SAGE_WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val, best_test = 0.0, 0.0
    for epoch in range(epochs):
        model.train()
        batch_seed = random.sample(train_nodes, k=min(64, len(train_nodes)))
        nodes_sorted, _, edge_index_sub, seed_mask = sample_subgraph(
            adj, batch_seed, sample_sizes, adaptive=adaptive)
        if not nodes_sorted:
            continue
        x_sub = data.x[nodes_sorted].to(DEVICE)
        y_sub = data.y[nodes_sorted].to(DEVICE)
        opt.zero_grad()
        logits = model(x_sub, edge_index_sub.to(DEVICE))
        loss = loss_fn(logits[seed_mask], y_sub[seed_mask])
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits_full = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
                preds = logits_full.argmax(dim=1)
                val_acc = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            if val_acc > best_val:  # select on validation, report test
                best_val, best_test = val_acc, test_acc
            if epoch % 50 == 0:
                print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                      f"Val {val_acc:.4f} | Test {test_acc:.4f}")

    return model, best_test


def sampling_rate_experiment(data, fixed_size=(10, 10), num_layers=2, epochs=200):
    """Fixed fan-out vs. the paper's degree-adaptive S_i = c + [N_i * w]."""
    print(f"\n-- Fixed sampling {list(fixed_size)} --")
    _, acc_fix = train_with_sampling(data, num_layers=num_layers,
                                     sample_sizes=fixed_size, epochs=epochs)
    print(f"\n-- Adaptive sampling S_i = {ADAPTIVE_C} + deg*{ADAPTIVE_W} "
          f"(cap {ADAPTIVE_CAP}) --")
    _, acc_adp = train_with_sampling(data, num_layers=num_layers,
                                     sample_sizes=fixed_size, adaptive=True,
                                     epochs=epochs)
    print(f"\nFixed sampling {list(fixed_size)} -> Test Acc: {acc_fix:.4f}")
    print(f"Adaptive sampling          -> Test Acc: {acc_adp:.4f}")
    return {"fixed": acc_fix, "adaptive": acc_adp}
