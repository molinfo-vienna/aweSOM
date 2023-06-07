import argparse
import logging
import os
import shutil
import sys
import torch

from operator import itemgetter
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from awesom.graph_neural_nets import GATv2
from awesom.pyg_dataset_creator import SOM
from awesom.utils import (
    EarlyStopping,
    plot_losses,
    save_individual,
    save_average,
    seed_everything
)

def run(
        device, 
        dataset, 
        loss, 
        hdim, 
        mha, 
        ns, 
        dropout, 
        epochs, 
        lr, 
        wd, 
        bs, 
        patience, 
        delta, 
        output_directory
    ):

    hyperparams = [hdim,mha,ns,dropout,lr,wd,bs]
    hyperparams_var_name = ["DimensionHiddenLayers",
                            "NumAttentionHeads",
                            "NegativeSlope",
                            "Dropout",
                            "LearningRate",
                            "WeightDecay",
                            "BatchSize"]

    logging.info("Start training...")

    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=False)

    y_preds = {}
    y_trues = {}
    mol_ids = {}

    for fold_num, (train_index, val_index) in enumerate(kf.split(dataset)):

        # Create results directory
        hyperparams_dir = ""
        for param in hyperparams:
            if type(param) == float:
                hyperparams_dir = hyperparams_dir + "_" + "{:.0e}".format(param)
            else:
                hyperparams_dir = hyperparams_dir + "_" + str(param)
        output_subdirectory = os.path.join(
            output_directory,
            hyperparams_dir
            + "_"
            + str(fold_num)
        )
        os.mkdir(os.path.join(os.getcwd(), output_subdirectory))

        # Initialize model
        model = GATv2(
            in_dim=dataset.num_features, 
            hdim=hdim, 
            edge_dim=dataset.num_edge_features, 
            heads=mha,
            negative_slope=ns,
            dropout=dropout, 
        ).to(device)

        # Split training and validation data for the current fold
        train_data = itemgetter(*train_index)(dataset)
        val_data = itemgetter(*val_index)(dataset)
        print(f"Training set: {len(train_data)} molecules.")
        print(f"Validation set: {len(val_data)} molecules.")

        #  Create training and validation data loader
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=bs, shuffle=True)

        """ ---------- Train Model ---------- """

        early_stopper = EarlyStopping(patience=patience, delta=delta, verbose=False)

        train_losses = []
        val_losses = []
        print('-' * 20)
        print(f'Training {fold_num+1}/{n_splits}')
        print('-' * 20)
        for epoch in tqdm(range(epochs)):
            final_num_epochs = epoch
            train_loss = model.train(train_loader, loss, lr, wd, device)
            train_losses.append(train_loss.item())
            val_loss, _, _, _, _ = model.test(val_loader, loss, device)
            val_losses.append(val_loss.item())
            early_stopper(val_loss)
            if early_stopper.early_stop:
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

        _, y_pred, mol_id, atom_id, y_true = model.test(val_loader, loss, device)

        y_preds[fold_num] = y_pred[:, 0]
        y_trues[fold_num] = y_true
        mol_ids[fold_num] = mol_id

        save_individual(
            output_directory,
            output_subdirectory,
            fold_num,
            "results.csv",
            final_num_epochs,
            val_loss.item(),
            y_pred[:, 0],
            y_true,
            mol_id,
            atom_id,
            hyperparams,
            hyperparams_var_name,
        )

    save_average(
        output_directory,
        "results_average.csv",
        y_preds,
        y_trues,
        mol_ids,
        hyperparams,
        hyperparams_var_name,
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
    parser.add_argument("-lf",
        "--loss",
        type=str,
        required=True,
        help="The loss function. Choose between \'BCE\', \'weighted_BCE\' and \'MCC_BCE\'.",    
    )
    parser.add_argument("-hd",
        "--dimensionHiddenLayers",
        type=int,
        required=True,
        help="The size of the hidden layers.",
    )
    parser.add_argument("-mha",
        "--multiHeadAttentions",
        type=int,
        required=True,
        help="The number of multi-head-attentions.",
    )
    parser.add_argument("-ns",
        "--negativeSlope",
        type=float,
        required=True,
        help="LeakyReLU angle of the negative slope.",
    )
    parser.add_argument("-do",
        "--dropout",
        type=float,
        required=True,
        help="Dropout probability.",
    )
    parser.add_argument("-e",
        "--epochs",
        type=int,
        required=True,
        help="The maximum number of training epochs.",
    )
    parser.add_argument("-lr",
        "--learningRate",
        type=float,
        required=True,
        help="Learning rate of the optimizer.",
    )
    parser.add_argument("-wd",
        "--weightDecay",
        type=float,
        required=True,
        help="Weight decay of the optimizer.",
    )
    parser.add_argument("-bs",
        "--batchSize", 
        type=int,
        required=True,
        help="Batch size.",
    )
    parser.add_argument("-p",
        "--patience",
        type=int,
        required=True,
        default=5,
        help="Early stopping: number of epochs with no improvement of the \
                        validation or test loss after which training will be stopped.",
    )
    parser.add_argument("-dt",
        "--delta",
        type=float,
        required=True,
        default=5,
        help="Early stopping: minimum change in the monitored quantity to qualify as an improvement, \
                        i.e. an absolute change of less than delta will count as no improvement.",  
    )
    parser.add_argument("-o",
        "--outputDirectory",
        type=str,
        required=True,
        help="The directory where the output is written."   
    )
    parser.add_argument("-hps",
        "--hyperParameterSearch",
        type=str,
        required=False, 
        default="False",
        help="Set to True when performing hyperparameter search via shell script. \
                When True, the prompt asking whether to append, overwrite or cancel \
                if the output folder already exists is deactivated and the results are \
                automatically appended."
    )
    parser.add_argument("-v",
        "--verbose",
        dest="verbosityLevel", 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the verbosity level of the logger - default is on INFO."
        )

    args = parser.parse_args()

    if args.hyperParameterSearch == "False":
        if os.path.exists(args.outputDirectory):
            overwrite = input(f"{args.outputDirectory} already exists. Append to existing data (a), overwrite directory (o) or cancel training (c)? [a/o/c] \n")
            if overwrite == "c":
                sys.exit()
            elif overwrite == "o":
                print("Overwriting directory...")
                shutil.rmtree(args.outputDirectory)
                os.makedirs(args.outputDirectory)
            elif overwrite == "a":
                print("Appending data...")
            else:
                logging.error("Training was terminated: incorrect command for dealing with existing output directory.")
                sys.exit()
        else:
            os.makedirs(args.outputDirectory)
    else:
        if not os.path.exists(args.outputDirectory):
            os.makedirs(args.outputDirectory)

    logging.basicConfig(filename= os.path.join(args.outputDirectory, 'logfile_train.log'), 
                    level=getattr(logging, args.verbosityLevel), 
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create/Load Custom PyTorch Geometric Dataset
    logging.info("Loading data")
    dataset = SOM(root=args.dir)
    logging.info("Data successfully loaded!")

    # Print dataset info
    print(f"Number of molecules in the data set: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}")
    print(f"Number of classes: {dataset.num_classes}")

    try:
        run(
        device,
        dataset,
        args.loss,
        args.dimensionHiddenLayers,
        args.multiHeadAttentions,
        args.negativeSlope,
        args.dropout,
        args.epochs,
        args.learningRate,
        args.weightDecay,
        args.batchSize,
        args.patience,
        args.delta,
        args.outputDirectory,
        )
    except Exception as e:
        logging.error("Training was terminated:", e)