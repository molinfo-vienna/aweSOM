import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from load_data.process_input_data import process_data
from load_data.pyg_dataset_creator import SOM
from models.mlp import MLP


def main():

    """
        -------------------------------------
        ------------Loading Data-------------
        ------------------------------------- 
    """

    # Process SDF input data to create PyTorch Geometric custom dataset
    #process_data(path='data/db_preprocessed.sdf')

    # Create/Load Custom PyTorch Geometric Dataset
    dataset = SOM(root='data')

    # Print dataset info
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Training/Validation/Test Split
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset = dataset[int(len(dataset)*0.9):]

    print(f'Training set: {len(train_dataset)} graphs.')
    print(f'Validation set: {len(val_dataset)} graphs.')
    print(f'Test set: {len(test_dataset)} graphs.')

    #  Data Loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    
    # Initialize MLP model
    model = MLP(in_dim=dataset.num_features, h_dim=16, out_dim=dataset.num_classes)
    print(model)

    def train(loader):
        model.train()
        criterion= torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        loss = 0
        for data in loader:
            optimizer.zero_grad()  # Clear gradients
            out = model(data.x)  # Perform a forward pass
            batch_loss = criterion(out, data.y)  # Compute loss function
            loss += batch_loss.item()
            batch_loss.backward()  # Derive gradients
            optimizer.step()  # Update parameters based on gradients
        return loss

    @torch.no_grad
    def test(loader):
        model.eval()
        predictions = []
        true_labels = []
        for data in loader:
            out = model(data.x)  # Perform a forward pass
            predictions.append(out.argmax(dim=1).float())   # Store the class with the highest probability for each data point
            true_labels.append(data.y)  # Store the true label of each data point in a more convenient format to compute F1 score
        true, pred = torch.cat(true_labels, dim=0).numpy(), torch.cat(predictions, dim=0).numpy()
        return f1_score(true, pred, average='micro')

    for epoch in range(10):
        training_loss = train(train_loader)
        val_f1 = test(val_loader)
        print(f'Epoch: {epoch}, Training Loss: {training_loss}, Validation F1-Score: {val_f1}.')


if __name__ == "__main__":
    main()