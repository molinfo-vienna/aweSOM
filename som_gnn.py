import argparse
import numpy as np
import os
import random
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import time
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily
from tqdm import tqdm

from src.process_input_data import process_data
from src.pyg_dataset_creator import SOM
from src.graph_neural_nets import GIN, GAT, train_oversampling, train, test
from src.utils import plot_losses, save_evaluation_results


def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", nargs="?", default="data/metaQSAR", help="the folder where the data is stored", type=str)
    parser.add_argument("data_name", nargs="?", default="metaQSAR.sdf", help="the name of the data file (must be a .sdf file)", type=str)
    parser.add_argument("output_directory", nargs="?", default="output/metaQSAR", help="the folder where the results will be stored", type=str)
    parser.add_argument("model_name", nargs="?", default= "GIN", help="the neural network that will be used: either \"GIN\" (Graph Isomorphism Network) or \"GAT\" (Graph Attention Network", type=str)
    parser.add_argument("h_dim", nargs="?", default=16, help="the size of the hidden layers", type=int)
    parser.add_argument("num_heads", nargs="?", default=0, help="the number of heads for the GAT model (please set to 0 when using GIN)", type=int)
    parser.add_argument("epochs", nargs="?", default=600, help="the number of training epochs", type=int)
    parser.add_argument("lr", nargs="?", default=0.0001, help="learning rate", type=float)
    parser.add_argument("wd", nargs="?", default=0.0001, help="weight decay", type=float)
    args = parser.parse_args()

    # Process SDF input data to create PyTorch Geometric custom dataset
    #process_data(args.data_directory, args.data_name)

    timestamp = int(time.time())
    output_subdirectory = os.path.join(args.output_directory, str(timestamp))
    os.mkdir(os.path.join(os.getcwd(), output_subdirectory))

    # Create/Load Custom PyTorch Geometric Dataset
    dataset = SOM(root=args.data_directory)

    # Print dataset info
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of node features: {dataset.num_node_features}')
    print(f'Number of edge features: {dataset.num_edge_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Compute homophily
    loader = DataLoader(dataset, batch_size=len(dataset))
    for data in loader: print(f'Homophily: {homophily(data.edge_index, data.y):.2f}')

    # Initialize model
    if args.model_name == "GIN":
        model = GIN(in_dim=dataset.num_features, h_dim=args.h_dim, edge_dim=dataset.num_edge_features).to(device)
    if args.model_name == "GAT":
        model = GAT(in_dim=dataset.num_features, h_dim=args.h_dim, num_heads=args.num_heads, edge_dim=dataset.num_edge_features).to(device)

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(dataset):
        # Training/Validation/Test Split
        train_dataset, val_dataset = dataset[train_index], dataset[test_index]
        print(f'Training set: {len(train_dataset)} molecules.')
        print(f'Validation set: {len(val_dataset)} molecules.')

        #  Data Loader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # Compute class weights of the training set:
        class_weights = 0
        total_num_instances = 0
        for data in train_loader:
            class_weights += compute_class_weight(class_weight='balanced', classes=np.unique(data.y), y=np.array(data.y)) * len(data.y)
            total_num_instances += len(data.y)
        class_weights /= total_num_instances


        """ ---------- Train Model ---------- """

        train_losses = []
        val_losses = []
        print('Training...')
        for _ in tqdm(range(args.epochs)):
            train_loss = train(model, train_loader, args.lr, args.wd, device)
            train_losses.append(train_loss.item())
            val_loss, _, _ ,_ = test(model, val_loader, device)
            val_losses.append(val_loss.item())
        print('Training done!')
        #torch.save(model.state_dict(), os.path.join(output_subdirectory, 'model.pt'))
        #plot_losses(train_losses, val_losses, path=os.path.join(output_subdirectory, 'loss.png'))

        """ ---------- Evaluate Model ---------- """

        #model.load_state_dict(torch.load(os.path.join(output_subdirectory, 'model.pt')))
        val_loss, val_pred, val_mol_ids, val_true = test(model, val_loader, device)
        try:
            val_pred_cv = np.append(val_pred_cv, val_pred[:,0])
        except:
            val_pred_cv = val_pred[:,0]
        try:
            val_mol_ids_cv = np.append(val_mol_ids_cv, val_mol_ids)
        except:
            val_mol_ids_cv = val_mol_ids
        try:
            val_true_cv = np.append(val_true_cv, val_true)
        except:
            val_true_cv = val_true

    save_evaluation_results(args.output_directory, output_subdirectory, timestamp, args.data_name, args.model_name, \
        args.h_dim, args.num_heads, args.epochs, args.lr, args.wd, \
        val_pred, val_mol_ids, val_true)

if __name__ == "__main__":
    main()
