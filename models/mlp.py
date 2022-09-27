import torch
import torch.nn.functional as F
from torch.nn import Linear

class MLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        torch.manual_seed(42)
        self.lin1 = Linear(in_dim, h_dim)
        self.lin2 = Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x