import argparse
import logging
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
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
    test_data,
    out,
    modelsPath,
):

    logging.info("Start testing...")

    y_preds = {}
    y_trues = {}
    opt_thresholds = []

    models = pd.read_csv(modelsPath).to_dict()
    print(f"Test set: {len(test_data)} molecules.")
    
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

        # Load test data
        test_loader = DataLoader(test_data, batch_size=models['Batch Size'][i], shuffle=True)

        # Apply model to test data
        _, y_pred, mol_id, atom_id, y_true = model.test(test_loader, device)

        for j, element in enumerate(zip(mol_id, atom_id)):
            y_preds.setdefault(element,[]).append(y_pred[:, 0][j])
            y_trues[element] = y_true[j]
        
        opt_thresholds.append(models['Optimal Threshold'][i])

    logging.info("Testing succesful!")
    
    logging.info("Saving results...")
    save_predict(
        out,
        y_preds,
        y_trues,
        opt_thresholds,
    )


if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Testing the model.")

    parser.add_argument("-d",
        "--dir",
        type=str,
        required=True,
        help="The directory where the input data is stored. There will be one training/evaluation and one test set."+
            " The test set size is based on the '--split' parameter.",    
    )
    parser.add_argument("-o",
        "--out",
        type=str,
        required=True,
        help="The directory where the output is written."   
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
        default='WARNING',
        help="Set the verbosity level of the logger - default is on WARNING."
        )

    args = parser.parse_args()

    logging.basicConfig(filename= os.path.join(args.out, 'logfile_test.log'), 
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
        test_data,
        args.out,
        args.modelsPath,
        )
    except Exception as e:
        logging.error("Testing was terminated:", e)