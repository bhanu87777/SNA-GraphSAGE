import networkx as nx
import numpy as np
from gensim.models import Word2Vec

def build_nx_from_edge_index(edge_index, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    return G

def generate_random_walks(G, num_walks=10, walk_length=40, seed=42):
    np.random.seed(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        random_order = nodes.copy()
        np.random.shuffle(random_order)
        for node in random_order:
            walk = [str(node)]
            current = node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = np.random.choice(neighbors)
                walk.append(str(current))
            walks.append(walk)
    return walks

def deepwalk_embedding(edge_index, num_nodes, dimensions=128, walks_per_node=10, walk_length=40, window=5, epochs=5, seed=42):
    G = build_nx_from_edge_index(edge_index, num_nodes)
    walks = generate_random_walks(G, num_walks=walks_per_node, walk_length=walk_length, seed=seed)
    w2v = Word2Vec(sentences=walks, vector_size=dimensions, window=window, min_count=0, sg=1, workers=4, epochs=epochs, seed=seed)
    emb = np.zeros((num_nodes, dimensions), dtype=np.float32)
    for n in range(num_nodes):
        key = str(n)
        emb[n] = w2v.wv[key] if key in w2v.wv else np.random.normal(size=(dimensions,))
    return emb
