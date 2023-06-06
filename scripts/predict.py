import argparse
import logging
import os
import pandas as pd
import shutil
import sys
import torch

from torch_geometric.loader import DataLoader

from som_gnn.graph_neural_nets import GAT, GATv2, GIN, GINNoPool, MF
from som_gnn.pyg_dataset_creator import SOM
from som_gnn.utils import (
    save_predict,
    seed_everything
)

def run(
    device,
    dataset,
    what_model,
    loss, 
    modelsDirectory, 
    numModels, 
    outputDirectory, 
):

    print("Start predicting...")
    logging.info("Start predicting...")

    y_preds = {}
    y_trues = {}
    opt_thresholds = []
    # Load info about trained models (hyperparameters, performances, directories)
    models_df = pd.read_csv(os.path.join(modelsDirectory, "results.csv"))
    # Sort models according to MCC
    models_df_ranked = models_df.sort_values(by=['MCC'], ascending=False)
    # Save metadata of the n best models to dict
    best_models = models_df_ranked.head(numModels).reset_index().to_dict()
    
    for i in range(len(best_models['ResultsFolder'])):

        # Initialize model
        if what_model == "GAT":
            model = GAT(
            in_dim=dataset.num_features, 
            hdim=best_models['DimensionHiddenLayers'][i],
            edge_dim=dataset.num_edge_features, 
            heads=best_models['NumAttentionHeads'][i],
            negative_slope=best_models['NegativeSlope'][i],
            dropout=best_models['Dropout'][i],
        ).to(device)
        if what_model == "GATv2":
            model = GATv2(
            in_dim=dataset.num_features, 
            hdim=best_models['DimensionHiddenLayers'][i],
            edge_dim=dataset.num_edge_features, 
            heads=best_models['NumAttentionHeads'][i],
            negative_slope=best_models['NegativeSlope'][i],
            dropout=best_models['Dropout'][i],
        ).to(device)
        elif what_model == "GIN":
            model = GIN(
                in_dim=dataset.num_features,
                hdim=best_models['DimensionHiddenLayers'][i],
                edge_dim=dataset.num_edge_features,
                dropout=best_models['Dropout'][i],
            ).to(device)
        elif what_model == "GINNoPool":
            model = GINNoPool(
                in_dim=dataset.num_features,
                hdim=best_models['DimensionHiddenLayers'][i],
                edge_dim=dataset.num_edge_features,
                dropout=best_models['Dropout'][i],
            ).to(device)
        elif what_model == "MF":
            model = MF(
            in_dim=dataset.num_features, 
            hdim=best_models['DimensionHiddenLayers'][i],
            max_degree=best_models['MaxDegree'][i],
        ).to(device)
        else:
            raise NotImplementedError(f"Invalid model: {what_model}")

        # Load saved model
        model.load_state_dict(torch.load(os.path.join(best_models['ResultsFolder'][i], "model.pt")))

        # Load data
        loader = DataLoader(dataset, batch_size=best_models['BatchSize'][i], shuffle=True)

        # Apply model to data
        _, y_pred, mol_id, atom_id, y_true = model.test(loader, loss, device)

        for index, molid_atomid_tuple in enumerate(zip(mol_id, atom_id)):
            y_preds.setdefault(molid_atomid_tuple,[]).append(y_pred[:, 0][index])
            y_trues[molid_atomid_tuple] = y_true[index]
        
        opt_thresholds.append(best_models['OptThreshold'][i])
        #opt_thresholds.append(0.5)
    
    logging.info("Saving results...")
    save_predict(
        outputDirectory, 
        y_preds,
        y_trues,
        opt_thresholds,
    )
    
    print("Predicting succesful!")
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
    parser.add_argument("-m",
        "--model",
        type=str,
        required=True,
        help="The chosen model. Choose between \'GAT\', \'GIN\' and \'MF\'.",    
    )
    parser.add_argument("-lf",
        "--loss",
        type=str,
        required=True,
        help="The loss function. Choose between \'BCE\', \'weighted_BCE\' and \'MCC_BCE\'.",    
    )
    parser.add_argument("-md",
        "--modelsDirectory",
        type=str,
        required=True,
        help="The directory where the trained models and the csv file containing \
            the trained models' hyperparameters and performance is stored.",
    )
    parser.add_argument("-n",
        "--numModels",
        type=int,
        required=True,
        help="The number of models to use for the ensemble classifier. \
            E.g. 10 will get the 10 best models from all models stored in the modelsDirectory.",
    )
    parser.add_argument("-o",
        "--outputDirectory",
        type=str,
        required=True,
        help="The directory where the output will be written."   
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
        args.model,
        args.loss, 
        args.modelsDirectory, 
        args.numModels, 
        args.outputDirectory, 
        )
    except Exception as e:
        logging.error("Predicting was terminated:", e)