import argparse
import logging
import os
import pandas as pd
import shutil
import sys
import torch

from collections import Counter
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score, 
)
from torch_geometric.loader import DataLoader
from tqdm import tqdm


from som_gnn.graph_neural_nets import GIN
from som_gnn.pyg_dataset_creator import SOM
from som_gnn.utils import (
    EarlyStopping,
    plot_roc_curve, 
    save_test,
    seed_everything, 
)


def run(
    device,
    train_data, 
    test_data, 
    out,
    modelsPath,
    epochs, 
    patience, 
    delta, 
):

    logging.info("Start testing...")

    models = pd.read_csv(modelsPath).to_dict()

    mccs = []
    precisions = []
    recalls = []

    num_testing_phases = 10
    for i in range(num_testing_phases):

        print('\n', '-' * 20)
        print(f'Testing Phase {i+1}/{num_testing_phases}')
        print('-' * 20)
        print("Retraining models...")
    
        # Build voting classifier based on the models stored in models
        # and store their predictions
        num_models = len(models['Results Folder'])
        for j in range(num_models):

            print('-' * 20)
            print(f'Training {j+1}/{num_models}')
            print('-' * 20)

            y_preds = {}
            y_trues = {}
            opt_thresholds = []

            # Initialize model
            model = GIN(
                in_dim=train_data.num_features,
                hdim=models['Dimension of Hidden Layers'][j],
                edge_dim=train_data.num_edge_features,
                dropout=models['Dropout'][j],
            ).to(device)

            # Load saved model
            #model.load_state_dict(torch.load(os.path.join(models['Results Folder'][j], "model.pt")))

            # Load data
            train_loader = DataLoader(train_data, batch_size=models['Batch Size'][j], shuffle=True)
            test_loader = DataLoader(test_data, batch_size=models['Batch Size'][j], shuffle=True)

            # Train model
            early_stopping = EarlyStopping(patience, delta)

            for _ in tqdm(range(epochs)):
                train_loss = model.train(train_loader, models['Learning Rate'][j], models['Weight Decay'][j], device)
                if early_stopping.early_stop(train_loss):
                    break

            # Apply trained model to test data
            _, y_pred, mol_id, atom_id, y_true = model.test(test_loader, device)

            # Compute best decision threshold
            opt_threshold = plot_roc_curve(y_true, y_pred, False)

            for j, element in enumerate(zip(mol_id, atom_id)):
                y_preds.setdefault(element,[]).append(y_pred[:, 0][j])
                y_trues[element] = y_true[j]
            
            opt_thresholds.append(opt_threshold)

        # Process individual predictions from voting classifier into voted predictions
        y_preds_bin = {}
        for threshold in opt_thresholds:
            for key in y_preds:
                for y_pred in y_preds[key]:
                    y_preds_bin.setdefault(key,[]).append(int(y_pred > threshold))
        y_preds_voted = {}
        for key in y_preds_bin:
            y_preds_voted[key] = Counter(y_preds_bin[key]).most_common()[0][0]

        # Compute metric of the predictions made by the current voting classifier
        # and append it to list of preictions
        mccs.append(matthews_corrcoef(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))
        precisions.append(precision_score(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))
        recalls.append(recall_score(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))

    save_test(
        mccs, 
        precisions, 
        recalls, 
        out, 
    )

    logging.info("Testing succesful!")
    
    logging.info("Saving results...")


if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Testing the model.")

    parser.add_argument("-d",
        "--dir",
        type=str,
        required=True,
        help="The directory where the training and test data is stored.",    
    )
    parser.add_argument("-o",
        "--out",
        type=str,
        required=True,
        help="The directory where the output will be written."   
    )
    parser.add_argument("-mp",
        "--modelsPath",
        type=str,
        required=True,
        help="The path of the csv file holding the metadata of the models chosen for the ensemble classifier.",
    )
    parser.add_argument("-e",
        "--epochs",
        type=int,
        required=True,
        help="the maximum number of training epochs",
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
    parser.add_argument("-v",
        "--verbose",
        dest="verbosityLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help="Set the verbosity level of the logger - default is on WARNING."
        )

    args = parser.parse_args()

    if os.path.exists(args.out):
        overwrite = input("Folder already exists. Overwrite? [y/n] \n")
        if overwrite == "y":
            shutil.rmtree(args.out)
            os.makedirs(args.out)
        if overwrite == "n":
            sys.exit()
    else:
        os.makedirs(args.out)

    logging.basicConfig(filename= os.path.join(args.out, 'logfile_predict.log'), 
                    level=getattr(logging, args.verbosityLevel), 
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create/Load Custom PyTorch Geometric Dataset
    logging.info("Loading data")
    train_data = SOM(root=os.path.join(args.dir, "train"))
    test_data = SOM(root=os.path.join(args.dir, "test"))
    logging.info("Data successfully loaded!")

    # Print dataset info
    print(f"Number of molecules in the training set: {len(train_data)}")
    print(f"Number of molecules in the test set: {len(test_data)}")
    print(f"Number of node features: {train_data.num_node_features}")
    print(f"Number of edge features: {train_data.num_edge_features}")
    print(f"Number of classes: {train_data.num_classes}")

    logging.info("Start testing")
    try:
        run(
        device,
        train_data,
        test_data, 
        args.out,
        args.modelsPath,
        args.epochs, 
        args.patience,
        args.delta
        )
    except Exception as e:
        logging.error("Testing was terminated:", e)