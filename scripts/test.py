import argparse
import logging
import os
import pandas as pd
import shutil
import sys
import torch

from collections import Counter
from operator import itemgetter
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score, 
)
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from som_gnn.graph_neural_nets import GIN
from som_gnn.pyg_dataset_creator import SOM
from som_gnn.utils import (
    plot_roc_curve, 
    save_test,
    seed_everything, 
)


def run(
    device, 
    test_data, 
    out, 
    modelsPath, 
):

    logging.info("Start testing...")

    models = pd.read_csv(modelsPath).to_dict()

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)

    mccs = []
    precisions = []
    recalls = []

    for fold_num, (_, index) in enumerate(kf.split(test_data)):

        print('-' * 20)
        print(f'Testing Phase {fold_num+1}/{n_splits}')

        # Get subset of test data
        sub_test_data = itemgetter(*index)(test_data)

        y_preds = {}
        y_trues = {}
        opt_thresholds = []
    
        # Build voting classifier based on the models stored in models
        # and store their predictions
        for j in range(len(models['Results Folder'])):

            # Initialize model
            model = GIN(
                in_dim=test_data.num_features,
                hdim=models['Dimension of Hidden Layers'][j],
                edge_dim=test_data.num_edge_features,
                dropout=models['Dropout'][j],
            ).to(device)

            # Load data
            data_loader = DataLoader(sub_test_data, batch_size=models['Batch Size'][j])

            # Load saved model and apply it to the test data current test data subset
            model.load_state_dict(torch.load(os.path.join(models['Results Folder'][j], "model.pt")))
            _, y_pred, mol_id, atom_id, y_true = model.test(data_loader, device)

            # Compute best decision threshold for current model and
            # append it to opt_thresholds list
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
        # and append it to list of predictions
        mccs.append(matthews_corrcoef(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))
        precisions.append(precision_score(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))
        recalls.append(recall_score(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))

    logging.info("Saving results...")
    save_test(
        mccs, 
        precisions, 
        recalls, 
        out, 
    )
    logging.info("Testing succesful!")


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
    parser.add_argument("-v",
        "--verbose",
        dest="verbosityLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the verbosity level of the logger - default is on INFO."
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
    test_data = SOM(args.dir)
    logging.info("Data successfully loaded!")

    # Print dataset info
    print(f"Number of molecules: {len(test_data)}")
    print(f"Number of node features: {test_data.num_node_features}")
    print(f"Number of edge features: {test_data.num_edge_features}")
    print(f"Number of classes: {test_data.num_classes}")

    try:
        run(
        device,
        test_data, 
        args.out,
        args.modelsPath,
        )
    except Exception as e:
        logging.error("Testing was terminated:", e)