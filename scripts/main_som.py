import argparse
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything

from som_gnn.pyg_dataset_creator import SOM
from som_gnn.runner import hp_optimization, predict
from som_gnn.utils import seed_everything


def main():

    seed_everything(42)

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")
    hp_optimization = subparser.add_parser("hp_optimization")
    predict = subparser.add_parser("predict")
    

    hp_optimization.add_argument(
        "--indir",
        type=str,
        required=True,
        help="the folder where the processed PYG dataset (input data) is stored",
    )
    hp_optimization.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="the folder where the results will be stored",
    )
    hp_optimization.add_argument(
        "--hdim", nargs="?",
        type=int,
        required=True,
        help="the size of the hidden layers",
    )
    hp_optimization.add_argument(
        "--dropout",
        type=float,
        required=True,
        help="dropout probability",
    )
    hp_optimization.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="the maximum number of training epochs",
    )
    hp_optimization.add_argument(
        "--lr",
        type=float,
        required=True,
        help="learning rate",
    )
    hp_optimization.add_argument(
        "--wd",
        type=float,
        required=True,
        help="weight decay",
    )
    hp_optimization.add_argument(
        "--bs", 
        type=int,
        required=True,
        help="batch size",
    )
    hp_optimization.add_argument(
        "--patience",
        type=int,
        required=True,
        default=5,
        help="early stopping: number of epochs with no improvement of the \
                        validation or test loss after which training will be stopped",
    )
    hp_optimization.add_argument(
        "--delta",
        type=float,
        required=True,
        default=5,
        help="early stopping: minimum change in the monitored quantity to qualify as an improvement, \
                        i.e. an absolute change of less than delta will count as no improvement",  
    )

    predict.add_argument(
        "--indir",
        type=str,
        required=True,
        help="the folder where the processed PYG dataset (input data) is stored",
    )
    predict.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="the folder where the results will be stored",
    )
    predict.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="the csv file holding the metadata of the models chosen for the ensemble classifier",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "hp_optimization":
        
        # Create/Load Custom PyTorch Geometric Dataset
        dataset = SOM(root=args.indir)

        # Print dataset info
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of edge features: {dataset.num_edge_features}")
        print(f"Number of classes: {dataset.num_classes}")

        # Training/Evaluation/Test Split
        train_val_data, test_data = train_test_split(
            dataset, test_size=0.1, random_state=42, shuffle=True
        )

        # train_val_molids = [item.mol_id[0].item() for item in train_val_data]
        # with open('data/train_val_molids.txt', 'w') as f:
        #     for item in train_val_molids:
        #         f.write("%s\n" % item)
        # test_molids = [item.mol_id[0].item() for item in test_data]
        # with open('data/test_molids.txt', 'w') as f:
        #     for item in test_molids:
        #         f.write("%s\n" % item)

        hp_optimization(
            device,
            dataset,
            train_val_data,
            args.outdir,
            args.hdim,
            args.dropout,
            args.epochs,
            args.lr,
            args.wd,
            args.bs,
            args.patience,
            args.delta,
        )

    elif args.command == "predict":

        # Create/Load Custom PyTorch Geometric Dataset
        dataset = SOM(root=args.indir)

        # Print dataset info
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of edge features: {dataset.num_edge_features}")
        print(f"Number of classes: {dataset.num_classes}")

        # Training/Evaluation/Test Split
        train_val_data, test_data = train_test_split(
            dataset, test_size=0.1, random_state=42, shuffle=True
        )

        predict(
            device,
            dataset,
            test_data,
            args.outdir,
            args.metadata,
        )

if __name__ == "__main__":
    main()