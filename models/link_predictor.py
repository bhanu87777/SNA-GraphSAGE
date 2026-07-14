"""Link prediction with a GraphSAGE encoder.

Edges are split into train/val/test with RandomLinkSplit (negatives sampled
1:1). The encoder embeds nodes using only training edges as message-passing
structure; a dot-product decoder scores candidate edges. Reported metrics are
ROC-AUC and Average Precision on the held-out test edges.
"""
import copy

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from models.gnn_zoo import GNN
from utils.data_loader import set_seeds


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.encoder = GNN("sage", in_channels, hidden_channels, out_channels,
                           num_layers=2, dropout=dropout)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    @staticmethod
    def decode(z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)  # dot product


@torch.no_grad()
def _eval_split(model, split):
    model.eval()
    z = model.encode(split.x, split.edge_index)
    scores = model.decode(z, split.edge_label_index).sigmoid().cpu().numpy()
    labels = split.edge_label.cpu().numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def run_link_prediction(data, hidden=128, out=64, epochs=200, lr=0.01,
                        device="cpu", seed=42, verbose=True):
    set_seeds(seed)
    # Negatives for training are NOT fixed here - a fresh negative sample is
    # drawn every epoch (below), which prevents the encoder from memorizing
    # one particular negative set and overfitting.
    transform = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
                                add_negative_train_samples=False)
    train_data, val_data, test_data = transform(data.cpu())
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    model = LinkPredictor(data.num_features, hidden, out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1),
            method="sparse")
        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat(
            [train_data.edge_label,
             train_data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

        logits = model.decode(z, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, edge_label)
        loss.backward()
        opt.step()

        val_auc, _ = _eval_split(model, val_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f}")

    model.load_state_dict(best_state)
    test_auc, test_ap = _eval_split(model, test_data)
    return {"test_auc": test_auc, "test_ap": test_ap, "val_auc": best_val_auc}
