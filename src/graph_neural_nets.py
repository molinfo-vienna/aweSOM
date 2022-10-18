import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout,LeakyReLU
from torch_geometric.nn import GINEConv, GATConv

def train(model, loader, class_weights, lr, weight_decay, device):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0
        total_num_instances = 0
        for data in loader:
            data = data.to(device)
            num_subsamplings = data.sampling_mask.shape[1]
            for i in range(num_subsamplings):
                _, out = model(data.x, data.edge_index, data.edge_attr)  # Perform a forward pass
                batch_loss = criterion( out[data.sampling_mask[:,i] == 1], 
                                        data.y[data.sampling_mask[:,i] == 1])  # Compute loss function
                loss += batch_loss * len(data.batch)
                total_num_instances += len(data.batch)
                optimizer.zero_grad()  # Clear gradients
                batch_loss.backward()  # Derive gradients
                optimizer.step()  # Update parameters based on gradients
        loss /= total_num_instances 
        return loss

@torch.no_grad()
def test(model, loader, class_weights, device):
    model.eval()
    #criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0
    total_num_instances = 0
    predictions = []
    true_labels = []
    for data in loader:
        data = data.to(device)
        _, out = model(data.x, data.edge_index, data.edge_attr)  # Perform a forward pass
        batch_loss = criterion(out, data.y.T.long())  # Compute loss function
        loss += batch_loss * len(data.batch)
        total_num_instances += len(data.batch)
        predictions.append(out.argmax(dim=1))   # Store the class with the highest probability for each data point
        true_labels.append(data.y)
    pred, true = torch.cat(predictions, dim=0).cpu().numpy(), torch.cat(true_labels, dim=0).cpu().numpy()
    loss /= total_num_instances
    return loss, pred, true


class GIN(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.conv1 = GINEConv(Sequential(Linear(in_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=0.5)
                                        ), edge_dim=4)
        self.conv2 = GINEConv(Sequential(Linear(h_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=0.5)
                                        ), edge_dim=4)
        self.conv3 = GINEConv(Sequential(Linear(h_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=0.5)
                                        ), edge_dim=4)
        self.lin1 = Linear(h_dim*3, h_dim*3)
        self.lin2 = Linear(h_dim*3, out_dim)

    def forward(self, x, edge_index, edge_attr):

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        #h_g = torch.nn.global_pooling(h[:i])

        # Classify
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)

        return h, F.softmax(h, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_heads):
        super().__init__()
        self.conv1 = GATConv(in_channels=in_dim, 
                            out_channels=h_dim, 
                            heads=num_heads, 
                            concat=True, 
                            negative_slope=0.2, 
                            dropout=0.3, 
                            add_self_loops=False, 
                            edge_dim=4, bias=True)
        self.conv2 = GATConv(in_channels=h_dim * num_heads, 
                            out_channels=h_dim, 
                            heads=num_heads, 
                            concat=True, 
                            negative_slope=0.2, 
                            dropout=0.3, 
                            add_self_loops=False, 
                            edge_dim=4, bias=True)
        self.conv3 = GATConv(in_channels=h_dim * num_heads, 
                            out_channels=h_dim, 
                            heads=num_heads, 
                            concat=True, 
                            negative_slope=0.2, 
                            dropout=0.3, 
                            add_self_loops=False, 
                            edge_dim=4, bias=True)
        self.lin1 = Linear(h_dim*num_heads*3, h_dim*num_heads*3)
        self.lin2 = Linear(h_dim*num_heads*3, out_dim)

    def forward(self, x, edge_index, edge_attr):

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        # Concatenate embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # h_g=torch.nn.global_pooling(h[0:45])

        # Classify
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)

        return h, F.softmax(h, dim=1)
        