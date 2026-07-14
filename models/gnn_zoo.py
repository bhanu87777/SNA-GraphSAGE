"""Unified multi-architecture GNN for node classification.

Supported architectures:
  - sage : GraphSAGE (Hamilton et al., 2017) with mean / max / sum aggregation
  - gcn  : Graph Convolutional Network (Kipf & Welling, 2017)
  - gat  : Graph Attention Network (Velickovic et al., 2018) - attention is the
           learned analogue of the paper's aggregator comparison
  - gin  : Graph Isomorphism Network (Xu et al., 2019) - the provably most
           expressive message-passing GNN referenced in the paper ([8])
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


def _make_conv(arch, in_dim, out_dim, aggr="mean", heads=1, concat_heads=True):
    if arch == "sage":
        return SAGEConv(in_dim, out_dim, aggr=aggr)
    if arch == "gcn":
        return GCNConv(in_dim, out_dim)
    if arch == "gat":
        return GATConv(in_dim, out_dim // heads if concat_heads else out_dim,
                       heads=heads, concat=concat_heads)
    if arch == "gin":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        return GINConv(mlp, train_eps=True)
    raise ValueError(f"Unknown architecture: {arch}")


class GNN(nn.Module):
    def __init__(self, arch, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, aggr="mean", heads=8):
        super().__init__()
        assert num_layers >= 1
        self.arch = arch
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(_make_conv(arch, in_channels, out_channels,
                                         aggr=aggr, heads=heads, concat_heads=False))
        else:
            self.convs.append(_make_conv(arch, in_channels, hidden_channels,
                                         aggr=aggr, heads=heads))
            for _ in range(num_layers - 2):
                self.convs.append(_make_conv(arch, hidden_channels, hidden_channels,
                                             aggr=aggr, heads=heads))
            self.convs.append(_make_conv(arch, hidden_channels, out_channels,
                                         aggr=aggr, heads=heads, concat_heads=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @torch.no_grad()
    def get_embeddings(self, x, edge_index):
        """Penultimate-layer activations (the learned node representation)."""
        self.eval()
        if len(self.convs) == 1:
            return self.convs[0](x, edge_index).detach()
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return x.detach()
