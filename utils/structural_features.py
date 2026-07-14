"""Classic social-network-analysis features computed per node.

These structural descriptors (centralities, clustering, k-core, ...) capture a
node's role in the network topology. Appended to the raw attribute features
they let the GNN reason about *position* in the network, not just content.
"""
import os

import networkx as nx
import numpy as np

from config import BETWEENNESS_SAMPLES

FEATURE_NAMES = [
    "degree",
    "log_degree",
    "clustering_coefficient",
    "pagerank",
    "betweenness_approx",
    "eigenvector_centrality",
    "core_number",
    "avg_neighbor_degree",
]


def compute_structural_features(edge_index, num_nodes, cache_path=None, seed=42):
    """Return a standardized (z-scored) [num_nodes, 8] float32 matrix."""
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    src = edge_index[0].cpu().numpy().tolist()
    dst = edge_index[1].cpu().numpy().tolist()
    G.add_edges_from(zip(src, dst))

    degree = np.array([G.degree(n) for n in range(num_nodes)], dtype=np.float64)
    clustering = np.array(list(nx.clustering(G).values()), dtype=np.float64)
    pagerank = np.array(list(nx.pagerank(G, alpha=0.85).values()), dtype=np.float64)
    betweenness = np.array(list(nx.betweenness_centrality(
        G, k=min(BETWEENNESS_SAMPLES, num_nodes), seed=seed).values()), dtype=np.float64)
    try:
        eig = nx.eigenvector_centrality_numpy(G)
        eigenvector = np.array([eig[n] for n in range(num_nodes)], dtype=np.float64)
    except Exception:
        eigenvector = np.zeros(num_nodes, dtype=np.float64)
    core = nx.core_number(G)
    core_number = np.array([core[n] for n in range(num_nodes)], dtype=np.float64)
    and_deg = nx.average_neighbor_degree(G)
    avg_neighbor_degree = np.array([and_deg[n] for n in range(num_nodes)], dtype=np.float64)

    feats = np.stack([
        degree,
        np.log1p(degree),
        clustering,
        pagerank,
        betweenness,
        eigenvector,
        core_number,
        avg_neighbor_degree,
    ], axis=1)

    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats = ((feats - mean) / std).astype(np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, feats)
    return feats
