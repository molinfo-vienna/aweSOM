import os
from operator import itemgetter
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from som_gnn.graph_neural_nets import GIN
from som_gnn.utils import (
    EarlyStopping,
    plot_losses,
    save_individual,
    save_average,
    save_predict
)


def _hp_optimization(
    device,
    dataset,
    train_val_data,
    output_directory,
    hdim,
    dropout,
    epochs,
    lr,
    wd,
    bs,
    patience,
    delta,
):

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


def _predict(
    device,
    dataset,
    test_data,
    outdir,
    metadata,
):
    y_preds = {}
    y_trues = {}
    opt_thresholds = []

    metadata = pd.read_csv(os.path.join(outdir, metadata)).to_dict()
    print(f"Test set: {len(test_data)} molecules.")
    
    for i in range(len(metadata['Results Folder'])):

        # Initialize model
        model = GIN(
            in_dim=dataset.num_features,
            hdim=metadata['Dimension of Hidden Layers'][i],
            edge_dim=dataset.num_edge_features,
            dropout=metadata['Dropout'][i],
        ).to(device)

        # Load saved model
        model.load_state_dict(torch.load(os.path.join(metadata['Results Folder'][i], "model.pt")))

        # Load test data
        test_loader = DataLoader(test_data, batch_size=metadata['Batch Size'][i], shuffle=True)

        # Apply model to test data
        _, y_pred, mol_id, atom_id, y_true = model.test(test_loader, device)

        for j, element in enumerate(zip(mol_id, atom_id)):
            y_preds.setdefault(element,[]).append(y_pred[:, 0][j])
            y_trues[element] = y_true[j]
        
        opt_thresholds.append(metadata['Optimal Threshold'][i])

    save_predict(
        outdir,
        y_preds,
        y_trues,
        opt_thresholds,
    )
