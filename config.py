import torch

SEED = 42
SEEDS = [42, 43, 44]  # multi-seed runs report mean +/- std
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = "Cora"  # Cora | CiteSeer | PubMed

# --- GNN training ---
GNN_HIDDEN = 128
GNN_DROPOUT = 0.5
GNN_LR = 0.01
GNN_WEIGHT_DECAY = 5e-4
GNN_EPOCHS = 300
GNN_PATIENCE = 30          # early stopping on validation accuracy
GAT_HEADS = 8

# --- legacy aliases (kept for older modules) ---
GRAPH_SAGE_HIDDEN = GNN_HIDDEN
GRAPH_SAGE_EMBED_DIM = GNN_HIDDEN
GRAPH_SAGE_LR = GNN_LR
GRAPH_SAGE_WEIGHT_DECAY = GNN_WEIGHT_DECAY
GRAPH_SAGE_EPOCHS = 200

# --- DeepWalk ---
DEEPWALK_DIM = 128
DEEPWALK_WALKS_PER_NODE = 10
DEEPWALK_WALK_LENGTH = 40
DEEPWALK_WINDOW = 5
DEEPWALK_EPOCHS = 5

LR_MAX_ITER = 1000

# --- structural (SNA) features ---
BETWEENNESS_SAMPLES = 128   # pivot count for approximate betweenness centrality

# --- degree-adaptive sampling: S_i = clip(c + w * deg(i), 1, cap) ---
ADAPTIVE_C = 5
ADAPTIVE_W = 0.5
ADAPTIVE_CAP = 25

# --- link prediction ---
LINKPRED_HIDDEN = 128
LINKPRED_OUT = 64
LINKPRED_EPOCHS = 200
LINKPRED_LR = 0.01

# --- community detection ---
TSNE_PERPLEXITY = 30

FIGURES_DIR = "figures"
RESULTS_DIR = "results"
