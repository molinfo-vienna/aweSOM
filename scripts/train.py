import argparse
import logging
import os
import torch

from operator import itemgetter
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from som_gnn.graph_neural_nets import GIN
from som_gnn.pyg_dataset_creator import SOM
from som_gnn.utils import (
    EarlyStopping,
    plot_losses,
    save_individual,
    save_average,
    seed_everything
)

def run(
        device, 
        dataset, 
        train_val_data, 
        hdim, 
        dropout, 
        epochs, 
        lr, 
        wd, 
        bs, 
        patience, 
        delta, 
        output_directory
    ):

    logging.info("Start training...")

    kf = KFold(n_splits=10, shuffle=False)

    y_preds = {}
    y_trues = {}
    mol_ids = {}

    for fold_num, (train_index, val_index) in enumerate(kf.split(train_val_data)):

        # Create results directory 
        output_subdirectory = os.path.join(
            output_directory,
            str(hdim)
            + "_"
            + "{:.0e}".format(dropout)
            + "_"
            + "{:.0e}".format(lr)
            + "_"
            + "{:.0e}".format(wd)
            + "_"
            + str(bs)
            + "_"
            + str(fold_num)
        )
        os.mkdir(os.path.join(os.getcwd(), output_subdirectory))

        # Initialize model
        model = GIN(
            in_dim=dataset.num_features,
            hdim=hdim,
            edge_dim=dataset.num_edge_features,
            dropout=dropout,
        ).to(device)

        # Split training and validation data for the current fold
        train_data = itemgetter(*train_index)(train_val_data)
        val_data = itemgetter(*val_index)(train_val_data)
        print(f"Training set: {len(train_data)} molecules.")
        print(f"Validation set: {len(val_data)} molecules.")

        #  Training and Validation Data Loader
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=bs, shuffle=True)

        """ ---------- Train Model ---------- """

        early_stopping = EarlyStopping(patience, delta)

        train_losses = []
        val_losses = []
        print("Training...")
        for epoch in tqdm(range(epochs)):
            final_num_epochs = epoch
            train_loss = model.train(train_loader, lr, wd, device)
            train_losses.append(train_loss.item())
            val_loss, _, _, _, _ = model.test(val_loader, device)
            val_losses.append(val_loss.item())
            if early_stopping.early_stop(val_loss):
                break
        torch.save(
            model.state_dict(),
            os.path.join(output_subdirectory, "model.pt"),
        )
        plot_losses(
            train_losses,
            val_losses,
            path=os.path.join(output_subdirectory, "loss.png"),
        )

        """ ---------- Validate Model ---------- """

        _, y_pred, mol_id, atom_id, y_true = model.test(val_loader, device)

        y_preds[fold_num] = y_pred[:, 0]
        y_trues[fold_num] = y_true
        mol_ids[fold_num] = mol_id

        save_individual(
            output_directory,
            output_subdirectory,
            fold_num,
            "results.csv",
            hdim,
            dropout,
            lr,
            wd,
            bs,
            final_num_epochs,
            val_loss.item(),
            y_pred[:, 0],
            y_true,
            mol_id,
            atom_id,
        )

    save_average(
        output_directory,
        "results_average.csv",
        hdim,
        dropout,
        lr,
        wd,
        bs,
        y_preds,
        y_trues,
        mol_ids,
    )

    logging.info("Training sucessful!")


if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Training the model.")

    parser.add_argument("-d",
        "--dir",
        type=str,
        required=True,
        help="The directory where the input data is stored.",    
    )
    parser.add_argument("-hd",
        "--hiddenLayersDimension",
        type=int,
        required=True,
        help="the size of the hidden layers",
    )
    parser.add_argument("-do",
        "--dropout",
        type=float,
        required=True,
        help="dropout probability",
    )
    parser.add_argument("-e",
        "--epochs",
        type=int,
        required=True,
        help="the maximum number of training epochs",
    )
    parser.add_argument("-lr",
        "--learningRate",
        type=float,
        required=True,
        help="learning rate",
    )
    parser.add_argument("-wd",
        "--weightDecay",
        type=float,
        required=True,
        help="weight decay",
    )
    parser.add_argument("-bs",
        "--batchSize", 
        type=int,
        required=True,
        help="batch size",
    )
    parser.add_argument("-p",
        "--patience",
        type=int,
        required=True,
        default=5,
        help="early stopping: number of epochs with no improvement of the \
                        validation or test loss after which training will be stopped",
    )
    parser.add_argument("-dt",
        "--delta",
        type=float,
        required=True,
        default=5,
        help="early stopping: minimum change in the monitored quantity to qualify as an improvement, \
                        i.e. an absolute change of less than delta will count as no improvement",  
    )
    parser.add_argument("-o",
        "--out",
        type=str,
        required=True,
        help="The directory where the output is written."   
    )
    parser.add_argument("-v",
        "--verbose",
        dest="verbosityLevel", 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help="Set the verbosity level of the logger - default is on WARNING."
        )

    args = parser.parse_args()

    logging.basicConfig(filename= os.path.join(args.out, 'logfile_train.log'), 
                    level=getattr(logging, args.verbosityLevel), 
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create/Load Custom PyTorch Geometric Dataset
    logging.info("Loading data...")
    dataset = SOM(root=args.dir)
    logging.info("Data successfully loaded!")

    # Print dataset info
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}")
    print(f"Number of classes: {dataset.num_classes}")

    # Training/Evaluation/Test Split
    logging.info("Splitting data...")
    train_val_data, test_data = train_test_split(
        dataset, test_size=0.1, random_state=42, shuffle=True
    )
    logging.info("Data sucessfully splitted!")

    try:
        run(
        device,
        dataset,
        train_val_data,
        args.hiddenLayersDimension,
        args.dropout,
        args.epochs,
        args.learningRate,
        args.weightDecay,
        args.batchSize,
        args.patience,
        args.delta,
        args.out,
        )
    except Exception as e:
        logging.error("Training was terminated:", e)