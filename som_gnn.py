import argparse
import torch

from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily

from src.process_input_data import process_data
from src.pyg_dataset_creator import SOM
from src.runner import hp_opt, testing


def main():

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("procedure", nargs="?", default="test", help="the tpe of procedure to run -- choose between \"process_data\", \"hp_opt\", \"test\"")
    parser.add_argument("data_directory", nargs="?", default="data/xenosite", help="the folder where the data is stored", type=str)
    parser.add_argument("data", nargs="?", default="xenosite.sdf", help="the name of the data file (must be a .sdf file)", type=str)
    parser.add_argument("output_directory", nargs="?", default="output/xenosite/test", help="the folder where the results will be stored", type=str)
    parser.add_argument("results_file_name", nargs="?", default="results_xenosite_test.csv", help="the name of the csv file to store the metrics", type=str)
    parser.add_argument("model", nargs="?", default= "GIN", help="the neural network that will be used: either \"GIN\" (Graph Isomorphism Network) or \"GAT\" (Graph Attention Network", type=str)
    parser.add_argument("h_dim", nargs="?", default=32, help="the size of the hidden layers", type=int)
    parser.add_argument("num_heads", nargs="?", default=0, help="the number of heads for the GAT model (set to 0 when using GIN)", type=int)
    parser.add_argument("epochs", nargs="?", default=300, help="the number of training epochs", type=int)
    parser.add_argument("lr", nargs="?", default=1e-3, help="learning rate", type=float)
    parser.add_argument("wd", nargs="?", default=1e-3, help="weight decay", type=float)
    parser.add_argument("batch_size", nargs="?", default=32, help="batch size", type=int)
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
        print(f'Validation set: {len(train_data)} molecules.')
        print(f'Test set: {len(test_data)} molecules.')

        # train_ids = []
        # for mol in train_data:
        #     train_ids.append(mol.mol_id[0].item())
        # with open("data/metaQSAR/metaQSAR_train_molids.txt", "w") as f:
        #     f.write(",".join(str(item) for item in train_ids))

        # val_ids = []
        # for mol in val_data:
        #     val_ids.append(mol.mol_id[0].item())
        # with open("data/metaQSAR/metaQSAR_val_molids.txt", "w") as f:
        #     f.write(",".join(str(item) for item in val_ids))

        # test_ids = []
        # for mol in test_data:
        #     test_ids.append(mol.mol_id[0].item())
        # with open("data/metaQSAR/metaQSAR_test_molids.txt", "w") as f:
        #     f.write(",".join(str(item) for item in test_ids))

        if args.procedure == "hp_opt":
            hp_opt(device, dataset, train_data, val_data, \
                args.output_directory, args.results_file_name, \
                    args.data, args.model, args.h_dim, \
                        args.num_heads, args.epochs, args.lr, args.wd, args.batch_size)

        if args.procedure == "test":
            testing(device, dataset, train_data, test_data, \
                    args.output_directory, args.results_file_name, \
                        args.data, args.model, args.h_dim, \
                            args.num_heads, args.epochs, args.lr, args.wd, args.batch_size)

if __name__ == "__main__":
    main()