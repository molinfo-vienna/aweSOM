import torch
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, LeakyReLU
from torch_geometric.nn import GINEConv, global_add_pool
from som_gnn.utils import MCC_BCE_Loss, MCC_Loss


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


    def train(self, loader, lr, wd, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        loss_function = MCC_BCE_Loss()
        loss = 0
        total_num_instances = 0
        for data in loader:
            data = data.to(device)
            out = self(data.x, data.edge_index, data.edge_attr, data.batch)
            batch_loss = loss_function(out[:, 0].to(float), data.y.to(float))
            loss += batch_loss * len(data.batch)
            total_num_instances += len(data.batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        loss /= total_num_instances
        return loss


    @torch.no_grad()
    def test(self, loader, device):
        loss_function = MCC_BCE_Loss()
        loss = 0
        total_num_instances = 0
        y_preds = []
        mol_ids = []
        atom_ids = []
        y_trues = []
        for data in loader:
            data = data.to(device)
            out = self(data.x, data.edge_index, data.edge_attr, data.batch)
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
