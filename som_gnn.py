import argparse
import ast
import numpy as np
import os
import torch
from distutils.util import strtobool
from ipywidgets import widgets
from rdkit.Chem import Draw, PandasTools
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily

from src.graph_neural_nets import GIN, GAT, test
from src.process_input_data import process_data
from src.pyg_dataset_creator import SOM
from src.runner import hp_opt, testing
from src.utils import seed_everything


def main():

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("procedure", nargs="?", default="hp_opt", help="the type of procedure to run: choose between \"process_data\", \"hp_opt\", \"test\", \"visualize\"")
    parser.add_argument("data_directory", nargs="?", default="data/xenosite", help="the folder where the data is stored", type=str)
    parser.add_argument("data", nargs="?", default="xenosite.sdf", help="the name of the data file (must be a .sdf file)", type=str)
    parser.add_argument("output_directory", nargs="?", default="output/xenosite", help="the folder where the results will be stored", type=str)
    parser.add_argument("results_file_name", nargs="?", default="results.csv", help="the name of the csv file to store the summarized results", type=str)
    parser.add_argument("model_name", nargs="?", default= "GIN", help="the neural network that will be used: either \"GIN\" (Graph Isomorphism Network) or \"GAT\" (Graph Attention Network", type=str)
    parser.add_argument("h_dim", nargs="?", default=32, help="the size of the hidden layers", type=int)
    parser.add_argument("dropout", nargs="?", default=0.2, help="dropout probability", type=float)
    parser.add_argument("num_heads", nargs="?", default=0, help="the number of heads for the GAT model (ignore when using GIN)", type=int)
    parser.add_argument("neg_slope", nargs="?", default=0, help="steepness of the negative slope for the GAT model (ignore when using GIN)", type=float)
    parser.add_argument("epochs", nargs="?", default=1000, help="the number of training epochs", type=int)
    parser.add_argument("lr", nargs="?", default=1e-3, help="learning rate", type=float)
    parser.add_argument("wd", nargs="?", default=1e-3, help="weight decay", type=float)
    parser.add_argument("batch_size", nargs="?", default=32, help="batch size", type=int)
    parser.add_argument("oversampling", nargs="?", default=False, help="whether to use oversampling technique or not: [True/False]", type=lambda x: bool(strtobool(x)))
    parser.add_argument("size_avg_window", nargs="?", default=10, help="early stopping: size of the interval taken into account to compute loss average", type=int)
    parser.add_argument("patience", nargs="?", default=5, help="early stopping: number of early stoppping evaluation phases with no improvement after which training will be stopped", type=int)
    parser.add_argument("delta", nargs="?", default=0., help="early stopping: minimal ratio to qualify as improvement", type=float)
    args = parser.parse_args()

    if args.procedure == "process_data":
        process_data(args.data_directory, args.data)  # process SDF input data to create PyTorch Geometric custom dataset

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # Training/Evaluation/Test Split
        train_val_data, test_data = train_test_split(dataset, test_size=1/10, random_state=42, shuffle=True)
        train_data, val_data = train_test_split(train_val_data, test_size=1/9, random_state=42, shuffle=True)
        print(f'Training set: {len(train_data)} molecules.')
        print(f'Validation set: {len(val_data)} molecules.')
        print(f'Test set: {len(test_data)} molecules.')

        if args.procedure == "hp_opt":
            hp_opt(device, dataset, train_data, val_data, \
                    args.output_directory, args.results_file_name, \
                        args.data, args.model_name, args.h_dim, args.dropout, \
                            args.num_heads, args.neg_slope, args.epochs, \
                                args.lr, args.wd, args.batch_size, args.oversampling, \
                                    args.size_avg_window, args.patience, args.delta)

        if args.procedure == "test":
            testing(device, dataset, train_data, test_data, \
                        args.output_directory, \
                            args.data, args.model_name, args.h_dim, args.dropout, \
                                args.num_heads, args.neg_slope, args.epochs, \
                                    args.lr, args.wd, args.batch_size, args.oversampling, \
                                        args.size_avg_window, args.patience, args.delta)

        if args.procedure == "visualize":
            if args.model_name == "GIN":
                model = GIN(in_dim=dataset.num_features, h_dim=args.h_dim, edge_dim=dataset.num_edge_features, dropout=args.dropout).to(device)
            if args.model_name == "GAT":
                model = GAT(in_dim=dataset.num_features, h_dim=args.h_dim, edge_dim=dataset.num_edge_features, num_heads=args.num_heads, neg_slope=args.neg_slope, dropout=args.dropout).to(device)
            # here you need to load an existing model
            model.load_state_dict(torch.load("output/xenosite/1670317775/model.pt"))

            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
            _, y_pred_test, mol_ids_test, y_true_test = test(model, test_loader, device)

            df = PandasTools.LoadSDF(os.path.join(args.data_directory, args.data), removeHs=True)
            df['soms'] = df['soms'].map(ast.literal_eval)

            ids = np.unique(mol_ids_test)

            def plot_molecules(i):
                id = ids[i]
                mask = id == mol_ids_test
                preds = y_pred_test[mask][:,0]
                trues = y_true_test[mask]
                ranking = np.argsort(-preds)
                mol = df._get_value(id, "ROMol")
                for atom, rank in zip(mol.GetAtoms(), ranking):
                    atom.SetProp("atomNote", str(rank+1))
                return Draw.MolToImage(mol, size=(500,500), highlightAtoms=list(map(int, list(np.where(trues==1)[0]))))

            slider = widgets.interact(plot_molecules, i=(0,len(ids),1))
            # here we would need to export the slider+images to an html file
            # I found this https://ipywidgets.readthedocs.io/en/latest/embedding.html#python-interface
            # But I haven't been able to implement it so far
            # you can have a look at mol.png for an example of the graphics generated by rdkit

if __name__ == "__main__":
    main()