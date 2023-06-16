import torch
from torch_geometric.nn import GATv2Conv, GINEConv, MFConv, TransformerConv, global_add_pool


__all__ = ["GATv2", "GIN", "MF", "TF"]


class GATv2(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 edge_dim, 
                 heads, 
                 negative_slope, 
                 dropout, 
                 n_conv_layers,
                 n_classifier_layers,
                 size_classify_layers):
        
        super(GATv2, self).__init__()

        self.conv1 = GATv2Conv(in_channels=in_channels,
                               out_channels=out_channels,
                               heads=heads,
                               negative_slope=negative_slope,
                               dropout=dropout,
                               edge_dim=edge_dim)

        self.conv =torch.nn. ModuleList([
            GATv2Conv(in_channels=out_channels*heads,
                      out_channels=out_channels,
                      heads=heads,
                      negative_slope=negative_slope,
                      dropout=dropout,
                      edge_dim=edge_dim)
            for _ in range(n_conv_layers)])

        self.classifier1 = torch.nn.Linear(out_channels*(n_conv_layers+1)*heads, size_classify_layers)

        self.classifier = torch.nn.ModuleList([
            torch.nn.Linear(size_classify_layers, size_classify_layers)
        for _ in range(n_classifier_layers)])

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)

        hs = []
        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)
    

class GIN(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 edge_dim, 
                 dropout, 
                 n_conv_layers,
                 n_classifier_layers,
                 size_classify_layers):
        
        super(GIN, self).__init__()

        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.BatchNorm1d(out_channels, out_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
                ), edge_dim=edge_dim)
            
        self.conv =torch.nn.ModuleList([
            GINEConv(
                torch.nn.Sequential(
                    torch.nn.Linear(out_channels, out_channels),
                    torch.nn.BatchNorm1d(out_channels, out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=dropout),
                    ), edge_dim=edge_dim)
            for _ in range(n_conv_layers)])

        self.classifier1 = torch.nn.Linear(out_channels*(n_conv_layers+1), size_classify_layers)

        self.classifier = torch.nn.ModuleList([
            torch.nn.Linear(size_classify_layers, size_classify_layers)
        for _ in range(n_classifier_layers)])

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)

        hs = []
        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class MF(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 max_degree, 
                 n_conv_layers,
                 n_classifier_layers,
                 size_classify_layers):
        
        super(MF, self).__init__()

        self.conv1 = MFConv(in_channels=in_channels,
                            out_channels=out_channels,
                            max_degree=max_degree)
            
        self.conv =torch.nn. ModuleList([
            MFConv(in_channels=out_channels,
                    out_channels=out_channels,
                    max_degree=max_degree)
            for _ in range(n_conv_layers)])

        self.classifier1 = torch.nn.Linear(out_channels*(n_conv_layers+1), size_classify_layers)

        self.classifier = torch.nn.ModuleList([
            torch.nn.Linear(size_classify_layers, size_classify_layers)
        for _ in range(n_classifier_layers)])

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)

        hs = []
        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class TF(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 edge_dim, 
                 heads, 
                 dropout, 
                 n_conv_layers,
                 n_classifier_layers,
                 size_classify_layers):
        
        super(TF, self).__init__()

        self.conv1 = TransformerConv(in_channels=in_channels,
                                     out_channels=out_channels,
                                     heads=heads,
                                     concat=True,
                                     beta=False,
                                     dropout=dropout,
                                     edge_dim=edge_dim,
                                     bias=True,
                                     root_weight=True)
            
        self.conv =torch.nn. ModuleList([
            TransformerConv(in_channels=out_channels*heads,
                            out_channels=out_channels,
                            heads=heads,
                            concat=True,
                            beta=False,
                            dropout=dropout,
                            edge_dim=edge_dim,
                            bias=True,
                            root_weight=True)
            for _ in range(n_conv_layers)])

        self.classifier1 = torch.nn.Linear(out_channels*(n_conv_layers+1), size_classify_layers)

        self.classifier = torch.nn.ModuleList([
            torch.nn.Linear(size_classify_layers, size_classify_layers)
        for _ in range(n_classifier_layers)])

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embeddings
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)

        hs = []
        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)
