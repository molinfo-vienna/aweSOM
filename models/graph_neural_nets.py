import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv

def train(model, loader, class_weights):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        loss = 0
        total_num_instances = 0
        for data in loader:
            optimizer.zero_grad()  # Clear gradients
            _, out = model(data.x, data.edge_index)  # Perform a forward pass
            batch_loss = criterion(out, data.y.T.long())  # Compute loss function
            loss += batch_loss * len(data.batch)
            total_num_instances += len(data.batch)
            batch_loss.backward()  # Derive gradients
            optimizer.step()  # Update parameters based on gradients
        loss /= total_num_instances
        return loss

@torch.no_grad()
def test(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    for data in loader:
        _, out = model(data.x, data.edge_index)  # Perform a forward pass
        predictions.append(out.argmax(dim=1))   # Store the class with the highest probability for each data point
        true_labels.append(data.y)
    pred, true = torch.cat(predictions, dim=0).numpy(), torch.cat(true_labels, dim=0).numpy()
    return pred, true


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
        