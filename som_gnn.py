import argparse
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily

from src.process_input_data import process_data
from src.pyg_dataset_creator import SOM
from src.runner import hp_opt, testing
from src.utils import seed_everything


def main():

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "procedure",
        nargs="?",
        default="testing",
        help='the type of procedure to run: choose between "process_data", "hp_opt", "testing"',
    )
    parser.add_argument(
        "data_directory",
        nargs="?",
        default="data_zaretzki",
        help="the folder where the data is stored",
        type=str,
    )
    parser.add_argument(
        "data",
        nargs="?",
        default="data.sdf",
        help="the name of the data file (must be .sdf file)",
        type=str,
    )
    parser.add_argument(
        "output_directory",
        nargs="?",
        default="../zaretzki_output_test/S1",
        help="the folder where the results will be stored",
        type=str,
    )
    parser.add_argument(
        "saved_models_summary_file",
        nargs="?",
        default="example.csv",
        help="the csv file holding metadata of the models chosen for the ensemble classifier",
        type=str,
    )
    parser.add_argument(
        "h_dim", nargs="?", default=64, help="the size of the hidden layers", type=int
    )
    parser.add_argument(
        "dropout", nargs="?", default=0.1, help="dropout probability", type=float
    )
    parser.add_argument(
        "epochs",
        nargs="?",
        default=1000,
        help="the maximum number of training epochs",
        type=int,
    )
    parser.add_argument("lr", nargs="?", default=1e-3, help="learning rate", type=float)
    parser.add_argument("wd", nargs="?", default=0, help="weight decay", type=float)
    parser.add_argument("batch_size", nargs="?", default=8, help="batch size", type=int)
    parser.add_argument(
        "patience",
        nargs="?",
        default=5,
        help="early stopping: number of epochs with no improvement of the \
                        validation or test loss after which training will be stopped",
        type=int,
    )
    parser.add_argument(
        "delta",
        nargs="?",
        default=5,
        help="early stopping: minimum change in the monitored quantity to qualify as an improvement, \
                        i.e. an absolute change of less than delta will count as no improvement",
        type=float,
    )
    args = parser.parse_args()

    if args.procedure == "process_data":
        # process SDF input data to create PyTorch Geometric custom dataset
        process_data(args.data_directory, args.data)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create/Load Custom PyTorch Geometric Dataset
        dataset = SOM(root=args.data_directory)

        # Print dataset info
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of edge features: {dataset.num_edge_features}")
        print(f"Number of classes: {dataset.num_classes}")

        # Compute homophily
        # loader = DataLoader(dataset, batch_size=len(dataset))
        # for data in loader:
        #     print(f"Homophily: {homophily(data.edge_index, data.y):.2f}")

        # Training/Evaluation/Test Split
        train_val_data, test_data = train_test_split(
            dataset, test_size=1 / 10, random_state=42, shuffle=True
        )

        # train_val_molids = [item.mol_id[0].item() for item in train_val_data]
        # with open('data/train_val_molids.txt', 'w') as f:
        #     for item in train_val_molids:
        #         f.write("%s\n" % item)
        # test_molids = [item.mol_id[0].item() for item in test_data]
        # with open('data/test_molids.txt', 'w') as f:
        #     for item in test_molids:
        #         f.write("%s\n" % item)

        if args.procedure == "hp_opt":
            hp_opt(
                device,
                dataset,
                train_val_data,
                args.output_directory,
                args.h_dim,
                args.dropout,
                args.epochs,
                args.lr,
                args.wd,
                args.batch_size,
                args.patience,
                args.delta,
            )

        if args.procedure == "testing":
            testing(
                device,
                dataset,
                test_data,
                args.output_directory,
                args.saved_models_summary_file,
            )


if __name__ == "__main__":
    main()