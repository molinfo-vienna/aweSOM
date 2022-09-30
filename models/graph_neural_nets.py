import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, h_dim)
        self.conv2 = GCNConv(h_dim, h_dim)
        self.conv3 = GCNConv(h_dim, h_dim)
        self.lin1 = Linear(h_dim*3, h_dim*3)
        self.lin2 = Linear(h_dim*3, out_dim)

    def forward(self, x, edge_index):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classify
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)

        return h, F.softmax(h, dim=1)

class GIN(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.conv1 = GINConv(Sequential(Linear(in_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        ReLU(),
                                        Linear(h_dim, h_dim),
                                        ReLU()))
        self.conv2 = GINConv(Sequential(Linear(h_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        ReLU(),
                                        Linear(h_dim, h_dim),
                                        ReLU()))
        self.conv3 = GINConv(Sequential(Linear(h_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        ReLU(),
                                        Linear(h_dim, h_dim),
                                        ReLU()))
        self.lin1 = Linear(h_dim*3, h_dim*3)
        self.lin2 = Linear(h_dim*3, out_dim)

    def forward(self, x, edge_index):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classify
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)

        return h, F.softmax(h, dim=1)