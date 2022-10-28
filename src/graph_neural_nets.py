import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout,LeakyReLU, AvgPool1d
from torch_geometric.nn import GINEConv, GATConv

def train(model, loader, lr, weight_decay, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()
    loss = 0
    total_num_instances = 0
    for data in loader:
        data = data.to(device)
        _, out = model(data.x, data.edge_index, data.edge_attr)  # Forward pass
        num_subsamplings = data.sampling_mask.shape[1]
        subsampling_losses = torch.zeros(num_subsamplings)
        for i in range(num_subsamplings):
            subsampling_losses[i] = loss_function( out[data.sampling_mask[:,i] == 1], data.y[data.sampling_mask[:,i] == 1])
        batch_loss = torch.sum(subsampling_losses) / subsampling_losses.size(dim=0)
        loss += batch_loss * len(data.batch)
        total_num_instances += len(data.batch)
        optimizer.zero_grad()  # Clear gradients
        batch_loss.backward()  # Derive gradients
        optimizer.step()  # Update parameters based on gradients
    loss /= total_num_instances 
    return loss


# def train(model, loader, class_weights, lr, weight_decay, device):
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
#     loss = 0
#     total_num_instances = 0
#     for data in loader:
#         data = data.to(device)
#         _, out = model(data.x, data.edge_index, data.edge_attr)  # Perform a forward pass
#         batch_loss = criterion(out, data.y.T.long())  # Compute loss function
#         loss += batch_loss * len(data.batch)
#         total_num_instances += len(data.batch)
#         optimizer.zero_grad()  # Clear gradients
#         batch_loss.backward()  # Derive gradients
#         optimizer.step()  # Update parameters based on gradients
#     loss /= total_num_instances
#     return loss

@torch.no_grad()
def test(model, loader, class_weights, device, threshold):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    loss = 0
    total_num_instances = 0
    predictions = []
    true_labels = []
    for data in loader:
        data = data.to(device)
        _, out = model(data.x, data.edge_index, data.edge_attr)
        batch_loss = loss_function(out, data.y.T.long())
        loss += batch_loss * len(data.batch)
        total_num_instances += len(data.batch)
        predictions.append((out[:,1] > threshold).int())
        true_labels.append(data.y)
    pred, true = torch.cat(predictions, dim=0).cpu().numpy(), torch.cat(true_labels, dim=0).cpu().numpy()
    loss /= total_num_instances
    return loss, pred, true


class GIN(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, edge_dim):
        super().__init__()
        self.conv1 = GINEConv(Sequential(Linear(in_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=0.3)
                                        ), edge_dim=edge_dim)
        self.conv2 = GINEConv(Sequential(Linear(h_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=0.3)
                                        ), edge_dim=edge_dim)
        self.conv3 = GINEConv(Sequential(Linear(h_dim, h_dim),
                                        BatchNorm1d(h_dim, h_dim),
                                        LeakyReLU(),
                                        Dropout(p=0.3)
                                        ), edge_dim=edge_dim)

        #self.pool = AvgPool1d(kernel_size=3, stride=1)
        self.lin = Sequential(Linear(h_dim*3, h_dim*3),
                              LeakyReLU(),
                              Linear(h_dim*3, out_dim))

    def forward(self, x, edge_index, edge_attr):

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        # Pooling
        #h_pool = self.pool(torch.stack((h1,h2,h3), dim=-1))

        # Concatenate embeddings
        #h = torch.cat((h1,h2, h3,torch.squeeze(h_pool)), dim=1)
        h = torch.cat((h1,h2,h3), dim=1)

        # Classify
        h = self.lin(h)

        return h, F.softmax(h, dim=1)


# class GAT(torch.nn.Module):
#     def __init__(self, in_dim, h_dim, out_dim, num_heads, edge_dim):
#         super().__init__()
#         self.conv1 = GATConv(in_channels=in_dim, 
#                             out_channels=h_dim, 
#                             heads=num_heads, 
#                             concat=True, 
#                             negative_slope=0.2, 
#                             dropout=0.3, 
#                             add_self_loops=False, 
#                             edge_dim=edge_dim, bias=True)
#         self.conv2 = GATConv(in_channels=h_dim * num_heads, 
#                             out_channels=h_dim, 
#                             heads=num_heads, 
#                             concat=True, 
#                             negative_slope=0.2, 
#                             dropout=0.3, 
#                             add_self_loops=False, 
#                             edge_dim=edge_dim, bias=True)
#         self.conv3 = GATConv(in_channels=h_dim * num_heads, 
#                             out_channels=h_dim, 
#                             heads=num_heads, 
#                             concat=True, 
#                             negative_slope=0.2, 
#                             dropout=0.3, 
#                             add_self_loops=False, 
#                             edge_dim=edge_dim, bias=True)
#         self.lin = Sequential(Linear(h_dim*num_heads*3, h_dim*num_heads*3),
#                               LeakyReLU(),
#                               Linear(h_dim*num_heads*3, out_dim))

#     def forward(self, x, edge_index, edge_attr):

#         # Node embeddings
#         h1 = self.conv1(x, edge_index, edge_attr)
#         h2 = self.conv2(h1, edge_index, edge_attr)
#         h3 = self.conv3(h2, edge_index, edge_attr)

#         # Concatenate embeddings
#         h = torch.cat((h1,h2,h3), dim=1)

#         # Classify
#         h = self.lin(h)

#         return h, F.softmax(h, dim=1)
        