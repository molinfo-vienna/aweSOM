import argparse
import logging
import os
import pandas as pd
import shutil
import sys
import torch

from torch_geometric.loader import DataLoader

from awesom.graph_neural_nets import GATv2, GIN, GINNA, GINPlus, GINE, GINENA, GINEPlus, MF, TF
from awesom.pyg_dataset_creator import SOM
from awesom.utils import seed_everything, save_predict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 64

def run(
    dataset,
    modelsDirectory, 
    outputDirectory, 
):

    print("Start predicting...")
    logging.info("Start predicting...")

    targets = {}
    predictions = {}
    opt_thresholds = []

    # Load data
    loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)

    tmp = pd.read_csv(os.path.join(modelsDirectory, "results_individual.csv"))
    
    for i in range(len(tmp)):

        # Read metadata
        info = {}
        with open(os.path.join(os.path.join(modelsDirectory, str(i+1)), "info.txt")) as f:
            for line in f:
                (key, val) = line.split()
                if key == "model":
                    info[key] = val
                else:
                    info[key] = float(val)

        # Initialize model
        if info["model"] == "GATv2":
            model = GATv2(in_channels=dataset.num_features, 
                          out_channels=int(info["out_channels"]), 
                          edge_dim=dataset.num_edge_features, 
                          heads=int(info["heads"]), 
                          negative_slope=info["negative_slope"], 
                          dropout=info["dropout"], 
                          n_conv_layers=int(info["n_conv_layers"]),
                          n_classifier_layers=int(info["n_classify_layers"]),
                          size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)
        elif info["model"] == "GIN":
            model = GIN(in_channels=dataset.num_features,  
                        out_channels=int(info["out_channels"]), 
                        dropout=info["dropout"], 
                        n_conv_layers=int(info["n_conv_layers"]),
                        n_classifier_layers=int(info["n_classify_layers"]),
                        size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)
        elif info["model"] == "GINNA":
            model = GINNA(in_channels=dataset.num_features,  
                          out_channels=int(info["out_channels"]), 
                          dropout=info["dropout"], 
                          n_conv_layers=int(info["n_conv_layers"]),
                          n_classifier_layers=int(info["n_classify_layers"]),
                          size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)
        elif info["model"] == "GIN+":
            model = GINPlus(in_channels=dataset.num_features,  
                            out_channels=int(info["out_channels"]), 
                            dropout=info["dropout"], 
                            n_conv_layers=int(info["n_conv_layers"]),
                            depth_conv_layers=int(info["depth_conv_layers"]),
                            n_classifier_layers=int(info["n_classify_layers"]),
                            size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)
        elif info["model"] == "GINE":
            model = GINE(in_channels=dataset.num_features,  
                        out_channels=int(info["out_channels"]), 
                        edge_dim=dataset.num_edge_features, 
                        dropout=info["dropout"], 
                        n_conv_layers=int(info["n_conv_layers"]),
                        n_classifier_layers=int(info["n_classify_layers"]),
                        size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)
        elif info["model"] == "GINENA":
            model = GINENA(in_channels=dataset.num_features,  
                        out_channels=int(info["out_channels"]), 
                        edge_dim=dataset.num_edge_features, 
                        dropout=info["dropout"], 
                        n_conv_layers=int(info["n_conv_layers"]),
                        n_classifier_layers=int(info["n_classify_layers"]),
                        size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)
        elif info["model"] == "GINE+":
            model = GINEPlus(in_channels=dataset.num_features,  
                        out_channels=int(info["out_channels"]), 
                        edge_dim=dataset.num_edge_features, 
                        dropout=info["dropout"], 
                        n_conv_layers=int(info["n_conv_layers"]),
                        depth_conv_layers=int(info["depth_conv_layers"]),
                        n_classifier_layers=int(info["n_classify_layers"]),
                        size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)
        elif info["model"] == "MF":
            model = MF(in_channels=dataset.num_features,  
                       out_channels=int(info["out_channels"]), 
                       max_degree=int(info["max_degree"]), 
                       n_conv_layers=int(info["n_conv_layers"]),
                       n_classifier_layers=int(info["n_classify_layers"]),
                       size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)
        elif info["model"] == "TF":
            model = TF(in_channels=dataset.num_features,  
                       out_channels=int(info["out_channels"]), 
                       edge_dim=dataset.num_edge_features, 
                       heads=int(info["heads"]), 
                       dropout=info["dropout"], 
                       n_conv_layers=int(info["n_conv_layers"]),
                       n_classifier_layers=int(info["n_classify_layers"]),
                       size_classify_layers=int(info["size_classify_layers"])).to(DEVICE)

        # Load saved state dictionary
        model.load_state_dict(torch.load(os.path.join(os.path.join(modelsDirectory, str(i+1)), "model.pt"), map_location=DEVICE))

        y_preds = []
        y_trues = []
        mol_ids = []
        atom_ids = []

        model.eval()
        for data in loader:
            data = data.to(DEVICE)
            if info["model"] in {"GIN", "GINNA", "GIN+"}:
                output = model(data.x, data.edge_index, data.batch)
            else:
                output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            y_preds.extend(output[:,0].tolist())
            y_trues.extend(data.y.tolist())
            mol_ids.extend(data.mol_id.tolist())
            atom_ids.extend(data.atom_id.tolist())

        for index, molid_atomid_tuple in enumerate(zip(mol_ids, atom_ids)):
            predictions.setdefault(molid_atomid_tuple,[]).append(y_preds[index])
            targets[molid_atomid_tuple] = y_trues[index]
        
        opt_thresholds.append(tmp['opt_threshold'][i])
    
    logging.info("Saving results...")
    save_predict(
        outputDirectory,
        targets,
        predictions,
        opt_thresholds,
    )
    
    print("Predicting succesful!")
    logging.info("Predicting succesful!")


if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Predicting SoMs...")

    parser.add_argument("-i",
        "--inputDirectory",
        type=str,
        required=True,
        help="The directory where the input data is stored.",    
    )
    parser.add_argument("-o",
        "--outputDirectory",
        type=str,
        required=True,
        help="The directory where the output will be written."   
    )
    parser.add_argument("-m",
        "--modelsDirectory",
        type=str,
        required=True,
        help="The directory where the trained models and the csv file containing \
            the trained models' hyperparameters and performance is stored.",
    )
    parser.add_argument("-v",
        "--verbose",
        dest="verbosityLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the verbosity level of the logger - default is on INFO."
        )

    args = parser.parse_args()

    if os.path.exists(args.outputDirectory):
        overwrite = input(f"{args.outputDirectory} already exists. Overwrite? [y/n] \n")
        if overwrite == "y":
            shutil.rmtree(args.outputDirectory)
            os.makedirs(args.outputDirectory)
        if overwrite == "n":
            sys.exit()
    else:
        os.makedirs(args.outputDirectory)

    logging.basicConfig(filename= os.path.join(args.outputDirectory, 'logfile_predict.log'), 
                    level=getattr(logging, args.verbosityLevel), 
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    # Create/Load Custom PyTorch Geometric Dataset
    logging.info("Loading data")
    dataset = SOM(root=args.inputDirectory)
    logging.info("Data successfully loaded!")

    # Print dataset info
    print(f"Number of molecules: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}")
    print(f"Number of classes: {dataset.num_classes}")

    try:
        run(
        dataset, 
        args.modelsDirectory, 
        args.outputDirectory, 
        )
    except Exception as e:
        logging.error("Predicting was terminated:", e)