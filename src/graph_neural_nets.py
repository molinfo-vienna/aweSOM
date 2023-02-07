import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, LeakyReLU
from torch_geometric.nn import GINEConv, global_add_pool


class MCC_Loss(torch.nn.Module):
    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = torch.sum(inputs * targets)
        TN = torch.sum((1 - inputs) * (1 - targets))
        FP = torch.sum((1 - targets) * inputs)
        FN = torch.sum(targets * (1 - inputs))
        MCC_loss = 1 - (TP * TN - FP * FN) / (
            torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
        return MCC_loss


class MCC_BCE_Loss(torch.nn.Module):
    def __init__(self):
        super(MCC_BCE_Loss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = torch.sum(inputs * targets)
        TN = torch.sum((1 - inputs) * (1 - targets))
        FP = torch.sum((1 - targets) * inputs)
        FN = torch.sum(targets * (1 - inputs))
        MCC_loss = 1 - (TP * TN - FP * FN) / (
            torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="sum")
        MCC_BCE_loss = MCC_loss + BCE_loss
        return MCC_BCE_loss


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hdim, edge_dim, dropout):
        super().__init__()
        self.conv1 = GINEConv(
            Sequential(
                Linear(in_dim, hdim),
                BatchNorm1d(hdim, hdim),
                LeakyReLU(),
                Dropout(p=dropout),
            ),
            edge_dim=edge_dim,
        )
        self.conv2 = GINEConv(
            Sequential(
                Linear(hdim, hdim),
                BatchNorm1d(hdim, hdim),
                LeakyReLU(),
                Dropout(p=dropout),
            ),
            edge_dim=edge_dim,
        )
        self.conv3 = GINEConv(
            Sequential(
                Linear(hdim, hdim),
                BatchNorm1d(hdim, hdim),
                LeakyReLU(),
                Dropout(p=dropout),
            ),
            edge_dim=edge_dim,
        )
        self.lin = Sequential(
            Linear(hdim * 4, hdim * 4), LeakyReLU(), Linear(hdim * 4, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        # Pooling
        h_pool = global_add_pool(h3, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3, h_pool_), dim=1)

        # Classify
        h = self.lin(h)

        return h


def train(model, loader, lr, wd, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_function = MCC_BCE_Loss()
    loss = 0
    total_num_instances = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        batch_loss = loss_function(out[:, 0].to(float), data.y.to(float))
        loss += batch_loss * len(data.batch)
        total_num_instances += len(data.batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    loss /= total_num_instances
    return loss


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    loss_function = MCC_BCE_Loss()
    loss = 0
    total_num_instances = 0
    y_preds = []
    mol_ids = []
    atom_ids = []
    y_trues = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        batch_loss = loss_function(out[:, 0].to(float), data.y.to(float))
        loss += batch_loss * len(data.batch)
        total_num_instances += len(data.batch)
        y_preds.append(torch.sigmoid(out))
        mol_ids.append(data.mol_id)
        atom_ids.append(data.atom_id)
        y_trues.append(data.y)
    y_pred, mol_id, atom_id, y_true = (
        torch.cat(y_preds, dim=0).cpu().numpy(),
        torch.cat(mol_ids, dim=0).cpu().numpy(),
        torch.cat(atom_ids, dim=0).cpu().numpy(),
        torch.cat(y_trues, dim=0).cpu().numpy(),
    )
    loss /= total_num_instances
    return loss, y_pred, mol_id, atom_id, y_true
