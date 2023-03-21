import argparse
import logging
import os
import pandas as pd
import shutil
import sys
import torch

from torch_geometric.loader import DataLoader

from som_gnn.graph_neural_nets import GIN
from som_gnn.pyg_dataset_creator import SOM
from som_gnn.utils import (
    save_predict,
    seed_everything
)

def run(
    device,
    dataset,
    out,
    modelsPath,
):

    logging.info("Start predicting...")

    y_preds = {}
    y_trues = {}
    opt_thresholds = []

    models = pd.read_csv(modelsPath).to_dict()
    
    for i in range(len(models['Results Folder'])):

        # Initialize model
        model = GIN(
            in_dim=dataset.num_features,
            hdim=models['Dimension of Hidden Layers'][i],
            edge_dim=dataset.num_edge_features,
            dropout=models['Dropout'][i],
        ).to(device)

        # Load saved model
        model.load_state_dict(torch.load(os.path.join(models['Results Folder'][i], "model.pt")))

        # Load data
        loader = DataLoader(dataset, batch_size=models['Batch Size'][i], shuffle=True)

        # Apply model to data
        _, y_pred, mol_id, atom_id, y_true = model.test(loader, device)

        for index, molid_atomid_tuple in enumerate(zip(mol_id, atom_id)):
            y_preds.setdefault(molid_atomid_tuple,[]).append(y_pred[:, 0][index])
            y_trues[molid_atomid_tuple] = y_true[index]
        
        opt_thresholds.append(models['Optimal Threshold'][i])
    
    logging.info("Saving results...")
    save_predict(
        out,
        y_preds,
        y_trues,
        opt_thresholds,
    )
    print("Done!")
    logging.info("Predicting succesful!")


if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Predicting SoMs...")

    parser.add_argument("-d",
        "--dir",
        type=str,
        required=True,
        help="The directory where the input data is stored.",    
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
    dataset = SOM(root=args.dir)
    logging.info("Data successfully loaded!")

    # Print dataset info
    print(f"Number of molecules: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}")
    print(f"Number of classes: {dataset.num_classes}")

    try:
        run(
        device,
        dataset,
        args.out,
        args.modelsPath,
        )
    except Exception as e:
        logging.error("Predicting was terminated:", e)