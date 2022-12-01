import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout,LeakyReLU
from torch_geometric.nn import GINEConv, GATConv, global_add_pool

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = torch.sum(inputs * targets)
        FP = torch.sum((1-targets) * inputs)
        FN = torch.sum(targets * (1-inputs))
        dice_loss = 1 - 2*TP/(2*TP + FP + FN)
        return dice_loss


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)       
        TP = torch.sum(inputs * targets)
        FP = torch.sum((1-targets) * inputs)
        FN = torch.sum(targets * (1-inputs))
        dice_loss = 1 - 2*TP/(2*TP + FP + FN)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='sum')
        diceBCE = BCE + dice_loss
        return diceBCE


class JaccardLoss(torch.nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = torch.sum(inputs * targets)  
        FP = torch.sum((1-targets) * inputs)
        FN = torch.sum(targets * (1-inputs))
        jaccard_loss = 1 - TP/(TP + FP + FN)    
        return jaccard_loss


class MCCLoss(torch.nn.Module):
    def __init__(self):
        super(MCCLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = torch.sum(inputs * targets)
        TN = torch.sum((1-inputs)*(1-targets))
        FP = torch.sum((1-targets) * inputs)
        FN = torch.sum(targets * (1-inputs))
        mcc_loss = 1 - (TP*TN - FP*FN)/(torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))  
        return mcc_loss


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean((output-target)**2)
        return loss


class TverskyLoss(torch.nn.Module):
    """
    For alpha = beta = 0.5, the Tversky index is equivalent to the Dice coefficient, a.k.a. F1 score.
    For alpha = beta = 1, the Tversky index is equal to the Tanimoto coefficient.
    For alpha + beta = 1, the Tversky index produces the F-beta score. Larger beta values weigh recall higher than precision.
    """
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = torch.sum(inputs * targets)
        FP = torch.sum((1-targets) * inputs)
        FN = torch.sum(targets * (1-inputs))
        Tversky = TP / (TP + self.alpha*FP + self.beta*FN)  
        return 1 - Tversky


class GIN(torch.nn.Module):
    def __init__(self, in_dim, h_dim, edge_dim, dropout):
        super().__init__()
        self.conv1 = GINEConv(Sequential(Linear(in_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=dropout)
                                        ), edge_dim=edge_dim)
        self.conv2 = GINEConv(Sequential(Linear(h_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=dropout)
                                        ), edge_dim=edge_dim)
        self.conv3 = GINEConv(Sequential(Linear(h_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=dropout)
                                        ), edge_dim=edge_dim)
        self.lin = Sequential(Linear(h_dim*4, h_dim*4),
                              LeakyReLU(),
                              Linear(h_dim*4, 1))

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
        h = torch.cat((h1,h2,h3,h_pool_), dim=1)

        # Classify
        h = self.lin(h)

        return h


class GAT(torch.nn.Module):
    def __init__(self, in_dim, h_dim, edge_dim, num_heads, neg_slope, dropout):
        super().__init__()
        self.conv1 = GATConv(in_channels=in_dim, 
                            out_channels=h_dim, 
                            heads=num_heads, 
                            concat=True, 
                            negative_slope=neg_slope, 
                            dropout=dropout, 
                            add_self_loops=False, 
                            edge_dim=edge_dim, bias=True)
        self.conv2 = GATConv(in_channels=h_dim * num_heads, 
                            out_channels=h_dim, 
                            heads=num_heads, 
                            concat=True, 
                            negative_slope=neg_slope, 
                            dropout=dropout, 
                            add_self_loops=False, 
                            edge_dim=edge_dim, bias=True)
        self.conv3 = GATConv(in_channels=h_dim * num_heads, 
                            out_channels=h_dim, 
                            heads=num_heads, 
                            concat=True, 
                            negative_slope=neg_slope, 
                            dropout=dropout, 
                            add_self_loops=False, 
                            edge_dim=edge_dim, bias=True)
        self.lin = Sequential(Linear(h_dim*num_heads*4, h_dim*num_heads*4),
                              LeakyReLU(),
                              Linear(h_dim*num_heads*4, 1))

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
        h = torch.cat((h1,h2,h3,h_pool_), dim=1)

        # Classify
        h = self.lin(h)

        return h


def train_oversampling(model, loader, lr, wd, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_function = torch.nn.BCEWithLogitsLoss()
    loss = 0
    total_num_instances = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        num_subsamplings = data.sampling_mask.shape[1]
        subsampling_losses = torch.zeros(num_subsamplings)
        for i in range(num_subsamplings):
            subsampling_losses[i] = loss_function(out[data.sampling_mask[:,i] == 1][:,0].to(float), data.y[data.sampling_mask[:,i] == 1].to(float))
        batch_loss = torch.sum(subsampling_losses) / subsampling_losses.size(dim=0)
        loss += batch_loss * len(data.batch)
        total_num_instances += len(data.batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    loss /= total_num_instances
    return loss

def train(model, loader, lr, wd, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_function = DiceBCELoss()
    loss = 0
    total_num_instances = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        batch_loss = loss_function(out[:,0].to(float), data.y.to(float))
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
    loss_function = DiceBCELoss()
    loss = 0
    total_num_instances = 0
    y_preds = []
    mol_ids = []
    y_trues = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        batch_loss = loss_function(out[:,0].to(float), data.y.to(float))
        loss += batch_loss * len(data.batch)
        total_num_instances += len(data.batch)
        y_preds.append(torch.sigmoid(out))
        mol_ids.append(data.mol_id)
        y_trues.append(data.y)
    y_pred, mol_id, y_true = torch.cat(y_preds, dim=0).cpu().numpy(), torch.cat(mol_ids, dim=0).cpu().numpy(), torch.cat(y_trues, dim=0).cpu().numpy()
    loss /= total_num_instances
    return loss, y_pred, mol_id, y_true
   