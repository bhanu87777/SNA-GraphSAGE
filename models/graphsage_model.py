import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # logits

    def get_embeddings(self, x, edge_index):
        out = x
        for i, conv in enumerate(self.convs):
            out = conv(out, edge_index)
            if i != len(self.convs) - 1:
                out = F.relu(out)
        if self.num_layers > 1:
            h = x
            for i, conv in enumerate(self.convs[:-1]):
                h = conv(h, edge_index)
                if i != len(self.convs[:-1]) - 1:
                    h = F.relu(h)
            return h.detach()
        else:
            return out.detach()

def train_full_batch(model, data, optimizer, epochs=200, device="cpu"):
    model.to(device)
    data = data.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        best_acc = max(best_acc, test_acc)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")
    return best_acc


# NOTE: superseded by models/gnn_zoo.py (GNN(arch="sage", ...)) which adds
# multi-architecture support and validation-based model selection.
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch_geometric.nn import SAGEConv

# class GraphSAGE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.dropout = dropout
#         if num_layers == 1:
#             self.convs.append(SAGEConv(in_channels, out_channels))
#         else:
#             self.convs.append(SAGEConv(in_channels, hidden_channels))
#             for _ in range(num_layers - 2):
#                 self.convs.append(SAGEConv(hidden_channels, hidden_channels))
#             self.convs.append(SAGEConv(hidden_channels, out_channels))

#     def forward(self, x, edge_index):
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             if i != len(self.convs)-1:
#                 x = F.relu(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#         return x

# def train_full_batch(model, data, optimizer, epochs=200, device="cpu"):
#     model, data = model.to(device), data.to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     best_test = 0.0
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         logits = model(data.x, data.edge_index)
#         loss_fn(logits[data.train_mask], data.y[data.train_mask]).backward()
#         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             pred = logits.argmax(1)
#             test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
#             best_test = max(best_test, test_acc)
#         if epoch % 50 == 0 or epoch == epochs-1:
#             print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Test: {test_acc:.4f}")
#     return best_test
