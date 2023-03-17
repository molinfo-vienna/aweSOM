import numpy as np
import torch
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, LeakyReLU
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.utils.class_weight import compute_class_weight
from som_gnn.utils import MCC_BCE_Loss, weighted_BCE_Loss


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
        #self.lin = Sequential(Linear(hdim * 4, hdim*4), LeakyReLU(), Linear(hdim * 4, 1))
        self.lin = Linear(hdim * 4, 1)

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

        return torch.sigmoid(h)


    def train(self, loader, lr, wd, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        #loss_function = torch.nn.BCELoss(reduction="sum")
        loss_function = weighted_BCE_Loss()
        running_loss = 0
        num_samples = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            #batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
            batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            running_loss += batch_loss * len(data.batch)
            num_samples += len(data.batch)
            batch_loss.backward()
            optimizer.step()
        return running_loss/num_samples


    @torch.no_grad()
    def test(self, loader, device):
        #loss_function = torch.nn.BCELoss(reduction="sum")
        loss_function = weighted_BCE_Loss()
        running_loss = 0
        num_samples = 0
        y_preds = []
        mol_ids = []
        atom_ids = []
        y_trues = []
        for data in loader:
            data = data.to(device)
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            #batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
            batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            running_loss += batch_loss * len(data.batch)
            num_samples += len(data.batch)
            y_preds.append(outputs)
            mol_ids.append(data.mol_id)
            atom_ids.append(data.atom_id)
            y_trues.append(data.y)
        y_pred, mol_id, atom_id, y_true = (
            torch.cat(y_preds, dim=0).cpu().numpy(),
            torch.cat(mol_ids, dim=0).cpu().numpy(),
            torch.cat(atom_ids, dim=0).cpu().numpy(),
            torch.cat(y_trues, dim=0).cpu().numpy(),
        )
        return running_loss/num_samples, y_pred, mol_id, atom_id, y_true
