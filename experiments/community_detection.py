"""Unsupervised community detection - a core social-network-analysis task.

Three approaches, all scored against the ground-truth classes with NMI/ARI:
  1. Louvain modularity maximization (pure structure)
  2. KMeans on raw node attributes (pure content)
  3. KMeans on trained GraphSAGE embeddings (structure + content)
"""
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score)

from config import (DEVICE, GNN_DROPOUT, GNN_EPOCHS, GNN_HIDDEN, GNN_LR,
                    GNN_PATIENCE, GNN_WEIGHT_DECAY, SEED)
from models.gnn_zoo import GNN
from utils.data_loader import set_seeds
from utils.train_utils import train_node_classifier
from utils.visualization import tsne_and_plot


def _score(y_true, y_pred):
    return (normalized_mutual_info_score(y_true, y_pred),
            adjusted_rand_score(y_true, y_pred))


def community_detection_experiment(dataset, data, epochs=GNN_EPOCHS, make_tsne=True):
    set_seeds(SEED)
    y = data.y.cpu().numpy()
    k = dataset.num_classes
    results = {}

    # 1) Louvain on the bare graph (structure only)
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.cpu().numpy()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    communities = nx.community.louvain_communities(G, seed=SEED)
    louvain_labels = np.zeros(data.num_nodes, dtype=int)
    for cid, members in enumerate(communities):
        for n in members:
            louvain_labels[n] = cid
    nmi, ari = _score(y, louvain_labels)
    results["Louvain (structure)"] = {"nmi": nmi, "ari": ari,
                                      "n_communities": len(communities)}
    print(f"Louvain: {len(communities)} communities | NMI {nmi:.4f} | ARI {ari:.4f}")

    # 2) KMeans on raw attributes (content only)
    km_raw = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    raw_labels = km_raw.fit_predict(data.x.cpu().numpy())
    nmi, ari = _score(y, raw_labels)
    results["KMeans on raw features"] = {"nmi": nmi, "ari": ari}
    print(f"KMeans raw features: NMI {nmi:.4f} | ARI {ari:.4f}")

    # 3) KMeans on GraphSAGE embeddings (structure + content)
    model = GNN("sage", dataset.num_node_features, GNN_HIDDEN,
                dataset.num_classes, num_layers=2, dropout=GNN_DROPOUT)
    res = train_node_classifier(model, data, epochs=epochs, lr=GNN_LR,
                                weight_decay=GNN_WEIGHT_DECAY,
                                patience=GNN_PATIENCE, device=DEVICE)
    emb = res["model"].get_embeddings(data.x.to(DEVICE),
                                      data.edge_index.to(DEVICE)).cpu().numpy()
    km_emb = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    emb_labels = km_emb.fit_predict(emb)
    nmi, ari = _score(y, emb_labels)
    results["KMeans on GraphSAGE emb."] = {"nmi": nmi, "ari": ari}
    print(f"KMeans GraphSAGE embeddings: NMI {nmi:.4f} | ARI {ari:.4f}")

    if make_tsne:
        f1 = tsne_and_plot(emb, y, "GraphSAGE_tSNE_true_labels")
        f2 = tsne_and_plot(emb, emb_labels, "GraphSAGE_tSNE_detected_communities")
        print("Saved t-SNE plots:", f1, "|", f2)

    return results
