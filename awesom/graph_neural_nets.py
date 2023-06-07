import numpy as np
import torch
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, LeakyReLU
from torch_geometric.nn import GATConv, GATv2Conv, GINEConv, MFConv, TransformerConv, global_add_pool
from sklearn.utils.class_weight import compute_class_weight

from awesom.utils import MCC_BCE_Loss, weighted_BCE_Loss, FocalLoss

__all__ = ["GAT", "Gatv2", "GIN", "GINNoPool", "MF", "TF"]


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hdim, edge_dim, heads, negative_slope, dropout):
        super().__init__()
        self.conv1 = GATConv(in_channels=in_dim,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             negative_slope=negative_slope,
                             dropout=dropout,
                             add_self_loops=True,
                             edge_dim=edge_dim,
                             fill_value="add",
                             bias=True)
        self.conv2 = GATConv(in_channels=hdim*heads,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             negative_slope=negative_slope,
                             dropout=dropout,
                             add_self_loops=True,
                             edge_dim=edge_dim,
                             fill_value="add",
                             bias=True)
        self.conv3 = GATConv(in_channels=hdim*heads,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             negative_slope=negative_slope,
                             dropout=dropout,
                             add_self_loops=True,
                             edge_dim=edge_dim,
                             fill_value="add",
                             bias=True)
        self.lin = Sequential(Linear(hdim*4*heads, hdim*3), 
                        LeakyReLU(), 
                        Linear(hdim*3, hdim*2), 
                        LeakyReLU(), 
                        Linear(hdim*2, 1))

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        # h_numpy = h1.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gat/bce/predict/5/h1.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h2.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gat/bce/predict/5/h2.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h3.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gat/bce/predict/5/h3.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Pooling
        h_pool = global_add_pool(h3, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3, h_pool_), dim=1)
        # h_numpy = h.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gat/bce/predict/5/h.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Classify
        h = self.lin(h)

        return torch.sigmoid(h)

    def train(self, loader, loss, lr, wd, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
            running_loss += batch_loss * len(data.batch)
            num_samples += len(data.batch)
            batch_loss.backward()
            optimizer.step()
        return running_loss/num_samples

    @torch.no_grad()
    def test(self, loader, loss, device):
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        y_preds = []
        mol_ids = []
        atom_ids = []
        y_trues = []
        for data in loader:
            data = data.to(device)
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
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


class GATv2(torch.nn.Module):
    def __init__(self, in_dim, hdim, edge_dim, heads, negative_slope, dropout):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=in_dim,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             negative_slope=negative_slope,
                             dropout=dropout,
                             add_self_loops=True,
                             edge_dim=edge_dim,
                             fill_value="add",
                             bias=True,
                             share_weights=False)
        self.conv2 = GATv2Conv(in_channels=hdim*heads,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             negative_slope=negative_slope,
                             dropout=dropout,
                             add_self_loops=True,
                             edge_dim=edge_dim,
                             fill_value="add",
                             bias=True,
                             share_weights=False)
        self.conv3 = GATv2Conv(in_channels=hdim*heads,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             negative_slope=negative_slope,
                             dropout=dropout,
                             add_self_loops=True,
                             edge_dim=edge_dim,
                             fill_value="add",
                             bias=True,
                             share_weights=False)
        self.lin = Sequential(Linear(hdim*4*heads, hdim*3), 
                        LeakyReLU(), 
                        Linear(hdim*3, hdim*2), 
                        LeakyReLU(), 
                        Linear(hdim*2, 1))

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

    def train(self, loader, loss, lr, wd, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
            running_loss += batch_loss * len(data.batch)
            num_samples += len(data.batch)
            batch_loss.backward()
            optimizer.step()
        return running_loss/num_samples

    @torch.no_grad()
    def test(self, loader, loss, device):
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        y_preds = []
        mol_ids = []
        atom_ids = []
        y_trues = []
        for data in loader:
            data = data.to(device)
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
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
        self.lin = Sequential(Linear(hdim*4, hdim*3), 
                        LeakyReLU(), 
                        Linear(hdim*3, hdim*2), 
                        LeakyReLU(), 
                        Linear(hdim*2, 1))
        # self.lin = Linear(hdim*4, 1)

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        # h_numpy = h1.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gin/bce/predict/5/h1.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h2.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gin/bce/predict/5/h2.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h3.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gin/bce/predict/5/h3.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Pooling
        h_pool = global_add_pool(h3, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3, h_pool_), dim=1)
        # h_numpy = h.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gin/bce/predict/5/h.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Classify
        h = self.lin(h)

        return torch.sigmoid(h)

    def train(self, loader, loss, lr, wd, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
            running_loss += batch_loss * len(data.batch)
            num_samples += len(data.batch)
            batch_loss.backward()
            optimizer.step()
        return running_loss/num_samples

    @torch.no_grad()
    def test(self, loader, loss, device):
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        y_preds = []
        mol_ids = []
        atom_ids = []
        y_trues = []
        for data in loader:
            data = data.to(device)
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
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
    

class GINNoPool(torch.nn.Module):
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
        self.lin = Sequential(Linear(hdim*3, hdim*3), 
                        LeakyReLU(), 
                        Linear(hdim*3, hdim*2), 
                        LeakyReLU(), 
                        Linear(hdim*2, 1))

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        # h_numpy = h1.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gin_no_pool/bce/predict/5/h1.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h2.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gin_no_pool/bce/predict/5/h2.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h3.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gin_no_pool/bce/predict/5/h3.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        # h_numpy = h.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gin_no_pool/bce/predict/5/h.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Classify
        h = self.lin(h)

        return torch.sigmoid(h)

    def train(self, loader, loss, lr, wd, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
            running_loss += batch_loss * len(data.batch)
            num_samples += len(data.batch)
            batch_loss.backward()
            optimizer.step()
        return running_loss/num_samples

    @torch.no_grad()
    def test(self, loader, loss, device):
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        y_preds = []
        mol_ids = []
        atom_ids = []
        y_trues = []
        for data in loader:
            data = data.to(device)
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
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


class MF(torch.nn.Module):
    def __init__(self, in_dim, hdim, max_degree):
        super().__init__()
        self.conv1 = MFConv(in_channels=in_dim,
                             out_channels=hdim,
                             max_degree=max_degree,
                             bias=True)
        self.conv2 = MFConv(in_channels=hdim,
                             out_channels=hdim,
                             max_degree=max_degree,
                             bias=True)
        self.conv3 = MFConv(in_channels=hdim,
                             out_channels=hdim,
                             max_degree=max_degree,
                             bias=True)
        self.lin = Sequential(Linear(hdim*4, hdim*3), 
                        LeakyReLU(), 
                        Linear(hdim*3, hdim*2), 
                        LeakyReLU(), 
                        Linear(hdim*2, 1))

    def forward(self, x, edge_index, batch):

        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        # h_numpy = h1.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/mf/bce/predict/5/h1.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h2.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/mf/bce/predict/5/h2.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h3.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/mf/bce/predict/5/h3.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Pooling
        h_pool = global_add_pool(h3, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3, h_pool_), dim=1)
        # h_numpy = h.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/mf/bce/predict/5/h.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Classify
        h = self.lin(h)

        return torch.sigmoid(h)

    def train(self, loader, loss, lr, wd, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = self(data.x, data.edge_index, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
            running_loss += batch_loss * len(data.batch)
            num_samples += len(data.batch)
            batch_loss.backward()
            optimizer.step()
        return running_loss/num_samples


    @torch.no_grad()
    def test(self, loader, loss, device):
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        y_preds = []
        mol_ids = []
        atom_ids = []
        y_trues = []
        for data in loader:
            data = data.to(device)
            outputs = self(data.x, data.edge_index, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
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
    

class TF(torch.nn.Module):
    def __init__(self, in_dim, hdim, edge_dim, heads, dropout):
        super().__init__()
        self.conv1 = TransformerConv(in_channels=in_dim,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             beta=False,
                             dropout=dropout,
                             edge_dim=edge_dim,
                             bias=True,
                             root_weight=True)
        self.conv2 = TransformerConv(in_channels=hdim*heads,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             beta=False,
                             dropout=dropout,
                             edge_dim=edge_dim,
                             bias=True,
                             root_weight=True)
        self.conv3 = TransformerConv(in_channels=hdim*heads,
                             out_channels=hdim,
                             heads=heads,
                             concat=True,
                             beta=False,
                             dropout=dropout,
                             edge_dim=edge_dim,
                             bias=True,
                             root_weight=True)
        self.lin = Sequential(Linear(hdim*4*heads, hdim*3), 
                        LeakyReLU(), 
                        Linear(hdim*3, hdim*2), 
                        LeakyReLU(), 
                        Linear(hdim*2, 1))

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        # h_numpy = h1.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gat/bce/predict/5/h1.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h2.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gat/bce/predict/5/h2.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)
        # h_numpy = h3.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gat/bce/predict/5/h3.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Pooling
        h_pool = global_add_pool(h3, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3, h_pool_), dim=1)
        # h_numpy = h.detach().cpu().resolve_conj().resolve_neg().numpy()
        # with open("output/metaQSAR/gat/bce/predict/5/h.txt", "ab") as f:
        #     np.savetxt(f, h_numpy)

        # Classify
        h = self.lin(h)

        return torch.sigmoid(h)

    def train(self, loader, loss, lr, wd, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
            running_loss += batch_loss * len(data.batch)
            num_samples += len(data.batch)
            batch_loss.backward()
            optimizer.step()
        return running_loss/num_samples

    @torch.no_grad()
    def test(self, loader, loss, device):
        if loss == "BCE":
            loss_function = torch.nn.BCELoss(reduction="sum")
        elif loss == "weighted_BCE":
            loss_function = weighted_BCE_Loss()
        elif loss == "MCC_BCE":
            loss_function = MCC_BCE_Loss()
        elif loss == "focal":
            loss_function = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss function: {loss}")
        running_loss = 0
        num_samples = 0
        y_preds = []
        mol_ids = []
        atom_ids = []
        y_trues = []
        for data in loader:
            data = data.to(device)
            outputs = self(data.x, data.edge_index, data.edge_attr, data.batch)
            if loss == "weighted_BCE":
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data.y.cpu()), y=np.array(data.y.cpu()))
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float), class_weights)
            else:
                batch_loss = loss_function(outputs[:, 0].to(float), data.y.to(float))
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
