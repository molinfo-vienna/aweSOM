import argparse
import logging
import numpy as np
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

from som_gnn.graph_neural_nets import GAT, GATv2, GIN, MF
from som_gnn.pyg_dataset_creator import SOM
from som_gnn.utils import (
    plot_roc_curve, 
    save_test,
    seed_everything, 
)


def run(
    device, 
    test_data, 
    what_model, 
    loss, 
    modelsDirectory, 
    numModels, 
    outputDirectory, 
):
    print("Start testing...")
    logging.info("Start testing...")

    # Load info about trained models (hyperparameters, performances, directories)
    models_df = pd.read_csv(os.path.join(modelsDirectory, "results.csv"))
    # Sort models according to MCC
    models_df_ranked = models_df.sort_values(by=['MCC'], ascending=False)
    # Save metadata of the n best models to dict
    best_models = models_df_ranked.head(numModels).reset_index().to_dict()

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)

    mccs = []
    precisions = []
    recalls = []
    top1s = []
    top2s = []

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
        for j in range(len(best_models['ResultsFolder'])):

            # Initialize model
            if what_model == "GAT":
                model = GAT(
                in_dim=test_data.num_features, 
                hdim=best_models['DimensionHiddenLayers'][j],
                edge_dim=test_data.num_edge_features, 
                heads=best_models['NumAttentionHeads'][j],
                negative_slope=best_models['NegativeSlope'][j],
                dropout=best_models['Dropout'][j],
            ).to(device)
            if what_model == "GATv2":
                model = GATv2(
                in_dim=test_data.num_features, 
                hdim=best_models['DimensionHiddenLayers'][j],
                edge_dim=test_data.num_edge_features, 
                heads=best_models['NumAttentionHeads'][j],
                negative_slope=best_models['NegativeSlope'][j],
                dropout=best_models['Dropout'][j],
            ).to(device)
            elif what_model == "GIN":
                model = GIN(
                    in_dim=test_data.num_features,
                    hdim=best_models['DimensionHiddenLayers'][j],
                    edge_dim=test_data.num_edge_features,
                    dropout=best_models['Dropout'][j],
                ).to(device)
            elif what_model == "MF":
                model = MF(
                in_dim=test_data.num_features, 
                hdim=best_models['DimensionHiddenLayers'][j],
                max_degree=best_models['MaxDegree'][j],
            ).to(device)
            else:
                raise NotImplementedError(f"Invalid model: {what_model}")

            # Load data
            data_loader = DataLoader(sub_test_data, batch_size=best_models['BatchSize'][j])

            # Load saved model and apply it to the test data current test data subset
            model.load_state_dict(torch.load(os.path.join(best_models['ResultsFolder'][j], "model.pt")))
            _, y_pred, mol_id, atom_id, y_true = model.test(data_loader, loss, device)

            # Compute best decision threshold for current model and
            # append it to opt_thresholds list
            opt_threshold = plot_roc_curve(y_true, y_pred, False)
            #opt_threshold = 0.5

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
        mccs.append(matthews_corrcoef(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))
        precisions.append(precision_score(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))
        recalls.append(recall_score(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values())))

        # Compute the averaged SoM probability per atom
        y_preds_avg = [sum(preds_list)/len(preds_list) for preds_list in y_preds.values()]

        # Compute top1 and top2 accuracies
        mol_ids = np.unique([a for a,b in y_trues.keys()])
        pred_top1 = []
        pred_top2 = []
        for mol_id in mol_ids: 
            mask = [a == mol_id for a,b in y_trues.keys()]
            idx = np.argpartition([y_preds_avg[i] for i, x in enumerate(mask) if x], -1)[-1:]
            if [list(y_trues.values())[i] for i, x in enumerate(mask) if x][idx[0]]:
                pred_top1.append(1)
            else:
                pred_top1.append(0)
            idx = np.argpartition([y_preds_avg[i] for i, x in enumerate(mask) if x], -2)[-2:]
            if ([list(y_trues.values())[i] for i, x in enumerate(mask) if x][idx[0]] or 
                [list(y_trues.values())[i] for i, x in enumerate(mask) if x][idx[1]]):
                pred_top2.append(1)
            else:
                pred_top2.append(0)
        top1s.append(np.sum(pred_top1) / len(mol_ids))
        top2s.append(np.sum(pred_top2) / len(mol_ids))

    logging.info("Saving results...")
    save_test(
        mccs, 
        precisions, 
        recalls, 
        top1s, 
        top2s, 
        outputDirectory, 
    )

    print("Testing succesful!")
    logging.info("Testing succesful!")


if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Testing the model.")

    parser.add_argument("-d",
        "--dataDirectory",
        type=str,
        required=True,
        help="The directory where the training and test data is stored.",    
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
    test_data = SOM(args.dataDirectory)
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
        args.model, 
        args.loss, 
        args.modelsDirectory, 
        args.numModels, 
        args.outputDirectory, 
        )
    except Exception as e:
        logging.error("Testing was terminated:", e)