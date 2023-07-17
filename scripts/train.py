import argparse
import csv
import itertools
import logging
import numpy as np
import optuna
import os
import torch

from operator import itemgetter
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from awesom.graph_neural_nets import (
    GATv2,
    GIN,
    GINNA,
    GINPlus,
    GINE,
    GINENA,
    GINEPlus,
    MF,
    TF,
)
from awesom.pyg_dataset_creator import SOM
from awesom.utils import (
    MCC_BCE_Loss,
    weighted_BCE_Loss,
    FocalLoss,
    EarlyStopping,
    plot_losses,
    seed_everything,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 20
DELTA = 0


def objective(
    trial: optuna.trial.Trial, train_loader: DataLoader, val_loader: DataLoader
) -> float:
    """Computes the objective value over an internal cross-validation fold.
    Args:
        trial (Optuna trial object): Optuna trial
        train_loader (PyG DataLoader object): training data loader
        val_loader (PyG DataLoader object): validation data loader
    Returns:
        rocauc (float): computed area under the receiver-operating-characteristic curve
    """
    out_channels = trial.suggest_int("out_channels", 8, 256)
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 5)
    n_classify_layers = trial.suggest_int("n_classify_layers", 1, 3)
    size_classify_layers = trial.suggest_int("size_classify_layers", 8, 256)

    if args.model == "GATv2":
        heads = trial.suggest_int("heads", 2, 8)
        negative_slope = trial.suggest_float("negative_slope", 0.1, 0.9)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model = GATv2(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            edge_dim=dataset.num_edge_features,
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    elif args.model == "GIN":
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model = GIN(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    elif args.model == "GINNA":
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model = GINNA(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    elif args.model == "GIN+":
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        depth_conv_layers = trial.suggest_int("depth_conv_layers", 1, 5)
        model = GINPlus(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            depth_conv_layers=depth_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    elif args.model == "GINE":
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model = GINE(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            edge_dim=dataset.num_edge_features,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    elif args.model == "GINENA":
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model = GINENA(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            edge_dim=dataset.num_edge_features,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    elif args.model == "GINE+":
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        depth_conv_layers = trial.suggest_int("depth_conv_layers", 1, 5)
        model = GINEPlus(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            edge_dim=dataset.num_edge_features,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            depth_conv_layers=depth_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    elif args.model == "MF":
        max_degree = trial.suggest_int("max_degree", 1, 20)
        model = MF(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            max_degree=max_degree,
            n_conv_layers=n_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    elif args.model == "TF":
        heads = trial.suggest_int("heads", 2, 8)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model = TF(
            in_channels=dataset.num_features,
            out_channels=out_channels,
            edge_dim=dataset.num_edge_features,
            heads=heads,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            n_classifier_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
        ).to(DEVICE)
    else:
        raise NotImplementedError(f"Invalid model: {args.model}")

    # Generate optimizer and scheduler
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    # Initialize early stopper
    early_stopper = EarlyStopping(patience=PATIENCE, delta=DELTA, verbose=False)

    for _ in tqdm(range(args.epochs)):
        validation_loss = []
        targets = []
        preds = []

        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()

            if args.model in {"GIN", "GINNA", "GIN+", "MF"}:
                output = model(data.x, data.edge_index, data.batch)
            else:
                output = model(data.x, data.edge_index, data.edge_attr, data.batch)

            if args.loss == "weighted_BCE":
                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(data.y.cpu()),
                    y=np.array(data.y.cpu()),
                )
                loss_value = loss_function(
                    output[:, 0].to(float), data.y.to(float), class_weights
                )
            else:
                loss_value = loss_function(output[:, 0].to(float), data.y.to(float))

            loss_value.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)

                if args.model in {"GIN", "GINNA", "GIN+", "MF"}:
                    output = model(data.x, data.edge_index, data.batch)
                else:
                    output = model(data.x, data.edge_index, data.edge_attr, data.batch)

                if args.loss == "weighted_BCE":
                    class_weights = compute_class_weight(
                        class_weight="balanced",
                        classes=np.unique(data.y.cpu()),
                        y=np.array(data.y.cpu()),
                    )
                    loss_value = loss_function(
                        output[:, 0].to(float), data.y.to(float), class_weights
                    )
                else:
                    loss_value = loss_function(output[:, 0].to(float), data.y.to(float))

                validation_loss.append(loss_value.item())
                targets.extend(data.y.tolist())
                preds.extend(output.tolist())

        rocauc = roc_auc_score(
            np.array(targets), np.array(list(itertools.chain(*preds)))
        )

        early_stopper(np.sum(np.array(validation_loss)))
        if early_stopper.early_stop:
            break

        scheduler.step(np.sum(np.array(validation_loss)))

    return rocauc


def objective_cv(trial: optuna.trial.Trial) -> float:
    """Computes the mean objective over the external cross-validation folds.
    Args:
        trial (Optuna trial object): Optuna trial
    Returns:
        mean_metric (float): mean optimized metric
    """
    fold = KFold(n_splits=args.numInternalCVFolds, shuffle=True, random_state=42)
    metric_list = []
    for fold_idx_int, (train_idx, val_idx) in enumerate(
        fold.split(range(len(train_val_data)))
    ):
        print(f"----- Internal CV fold {fold_idx_int+1}/{args.numInternalCVFolds}")

        train_data = itemgetter(*train_idx)(train_val_data)
        val_data = itemgetter(*val_idx)(train_val_data)

        train_loader = DataLoader(
            train_data,
            batch_size=args.batchSize,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batchSize,
            shuffle=True,
        )

        metric = objective(trial, train_loader, val_loader)
        metric_list.append(metric)

    mean_metric = np.mean(metric_list)

    return mean_metric


if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser("Training the model.")

    parser.add_argument(
        "-i",
        "--inputDirectory",
        type=str,
        required=True,
        help="The directory where the input data is stored.",
    )
    parser.add_argument(
        "-o",
        "--outputDirectory",
        type=str,
        required=True,
        help="The directory where the results are written.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The desired model architecture. Choose between 'GATv2', 'GIN', 'MF' and 'TF'.",
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        required=True,
        help="The desired loss function. Choose between 'BCE', 'weighted_BCE', 'MCC_BCE' and 'focal'.",
    )
    parser.add_argument(
        "-b",
        "--batchSize",
        type=int,
        required=True,
        help="The batch size of the data loader.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=True,
        help="The maximum number of training epochs.",
    )
    parser.add_argument(
        "-nt",
        "--nTrials",
        type=int,
        required=True,
        help="The number of Optuna trials.",
    )
    parser.add_argument(
        "-nif",
        "--numInternalCVFolds",
        type=int,
        required=True,
        help="The number of internal cross-validation folds.",
    )
    parser.add_argument(
        "-nef",
        "--numExternalCVFolds",
        type=int,
        required=True,
        help="The number of external cross-validation folds.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosityLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the verbosity level of the logger - default is on INFO.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    logging.basicConfig(
        filename=os.path.join(args.outputDirectory, "logfile_train.log"),
        level=getattr(logging, args.verbosityLevel),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    loss_function: torch.nn.modules.loss._Loss

    # Retrieve user-defined hyperparameters
    if args.loss == "BCE":
        loss_function = torch.nn.BCELoss(reduction="sum")
    elif args.loss == "weighted_BCE":
        loss_function = weighted_BCE_Loss()
    elif args.loss == "MCC_BCE":
        loss_function = MCC_BCE_Loss()
    elif args.loss == "focal":
        loss_function = FocalLoss()
    else:
        raise NotImplementedError(f"Invalid loss function: {args.loss}")

    # Create/Load Custom PyTorch Geometric Dataset
    logging.info("Loading data")
    dataset = SOM(root=args.inputDirectory)
    logging.info("Data successfully loaded!")

    # Print dataset info
    print(f"Number of molecules in the data set: {len(dataset)}")
    print(f"Dimension node features: {dataset.num_node_features}")
    print(f"Dimension edge features: {dataset.num_edge_features}")
    print(f"Number of classes: {dataset.num_classes}")

    with open(
        os.path.join(args.outputDirectory, "results_individual.csv"),
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "opt_threshold",
                "mcc",
                "precision",
                "recall",
                "auc_pr",
                "roc_auc",
            )
        )

    opt_threshold_list = []
    mcc_list = []
    precision_list = []
    recall_list = []
    auc_pr_list = []
    rocauc_list = []

    fold = KFold(n_splits=args.numExternalCVFolds, shuffle=True, random_state=42)
    for fold_idx_ext, (train_val_idx, test_idx) in enumerate(fold.split(dataset)):
        print(f"External CV fold {fold_idx_ext+1}/{args.numExternalCVFolds}")
        logging.info(
            f"Starting external CV fold {fold_idx_ext+1}/{args.numExternalCVFolds}"
        )

        if not os.path.exists(
            os.path.join(args.outputDirectory, str(fold_idx_ext + 1))
        ):
            os.mkdir(os.path.join(args.outputDirectory, str(fold_idx_ext + 1)))

        # Create optuna study
        study = optuna.create_study(
            storage="sqlite:///" + args.outputDirectory + "/storage.db",
            study_name="fold-" + str(fold_idx_ext),
            direction="maximize",
            load_if_exists=True,
        )

        # Get data
        train_val_data = itemgetter(*train_val_idx)(dataset)
        test_data = itemgetter(*test_idx)(dataset)

        #####################################################################
        #          tune hyperparameters on training/validation data
        #####################################################################

        logging.info("Searching for optimal hyperparameters...")
        study.optimize(objective_cv, n_trials=args.nTrials)

        best_trial = study.best_trial

        print("Best trial:")
        print("  Value: ", best_trial.value)
        logging.info(
            f"Best MCC for {fold_idx_ext+1}/{args.numExternalCVFolds}: {study.best_trial.value}"
        )
        logging.info(
            f"Best hyperparameters fold {fold_idx_ext+1}/{args.numExternalCVFolds}:"
        )
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key} {value}")
            logging.info(f"    {key} {value}")

        with open(
            os.path.join(args.outputDirectory, str(fold_idx_ext + 1) + "/info.txt"), "w"
        ) as f:
            f.write(f"model {args.model}\n")
            for key, value in best_trial.params.items():
                f.write(f"{key} {value}\n")

        #####################################################################
        # retrain model with best hyperparameters and test model on test data
        #####################################################################

        logging.info("Retraining model with best hyperparameters ...")
        print("Retraining model with best hyperparameters ...")

        model: torch.nn.Module

        if args.model == "GATv2":
            model = GATv2(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                edge_dim=dataset.num_edge_features,
                heads=best_trial.params["heads"],
                negative_slope=best_trial.params["negative_slope"],
                dropout=best_trial.params["dropout"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        elif args.model == "GIN":
            model = GIN(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                dropout=best_trial.params["dropout"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        elif args.model == "GINNA":
            model = GINNA(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                dropout=best_trial.params["dropout"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        elif args.model == "GIN+":
            model = GINPlus(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                dropout=best_trial.params["dropout"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                depth_conv_layers=best_trial.params["depth_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        elif args.model == "GINE":
            model = GINE(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                edge_dim=dataset.num_edge_features,
                dropout=best_trial.params["dropout"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        elif args.model == "GINENA":
            model = GINENA(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                edge_dim=dataset.num_edge_features,
                dropout=best_trial.params["dropout"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        elif args.model == "GINE+":
            model = GINEPlus(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                edge_dim=dataset.num_edge_features,
                dropout=best_trial.params["dropout"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                depth_conv_layers=best_trial.params["depth_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        elif args.model == "MF":
            model = MF(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                max_degree=best_trial.params["max_degree"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        elif args.model == "TF":
            model = TF(
                in_channels=dataset.num_features,
                out_channels=best_trial.params["out_channels"],
                edge_dim=dataset.num_edge_features,
                heads=best_trial.params["heads"],
                dropout=best_trial.params["dropout"],
                n_conv_layers=best_trial.params["n_conv_layers"],
                n_classifier_layers=best_trial.params["n_classify_layers"],
                size_classify_layers=best_trial.params["size_classify_layers"],
            ).to(DEVICE)
        else:
            raise NotImplementedError(f"Invalid model: {args.model}")

        # Resplit training/val data randomly into training (90%) and validation (10%) sets
        # the randomly chosen validation is used to determine the optimal number of training
        # epochs via early stopping

        train_data, val_data = train_test_split(
            train_val_data, test_size=0.2, shuffle=True, random_state=None
        )

        train_loader = DataLoader(
            train_data,
            batch_size=args.batchSize,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batchSize,
            shuffle=True,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=args.batchSize,
            shuffle=True,
        )

        # Generate the optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=best_trial.params["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )

        # Initialize early stopper
        early_stopper = EarlyStopping(patience=PATIENCE, delta=DELTA, verbose=False)

        # Retrain model
        training_losses = []
        validation_losses = []

        for epoch in tqdm(range(args.epochs)):
            model.train()
            batch_training_losses = []
            num_training_samples = 0
            for data in train_loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()

                if args.model in {"GIN", "GINNA", "GIN+", "MF"}:
                    output = model(data.x, data.edge_index, data.batch)
                else:
                    output = model(data.x, data.edge_index, data.edge_attr, data.batch)

                if args.loss == "weighted_BCE":
                    class_weights = compute_class_weight(
                        class_weight="balanced",
                        classes=np.unique(data.y.cpu()),
                        y=np.array(data.y.cpu()),
                    )
                    batch_training_loss = loss_function(
                        output[:, 0].to(float), data.y.to(float), class_weights
                    )
                else:
                    batch_training_loss = loss_function(
                        output[:, 0].to(float), data.y.to(float)
                    )

                batch_training_losses.append(batch_training_loss.item())
                num_training_samples += len(data.batch)
                batch_training_loss.backward()
                optimizer.step()

            training_losses.append(
                np.sum(np.array(batch_training_losses)) / num_training_samples
            )

            model.eval()
            with torch.no_grad():
                batch_validation_losses = []
                num_validation_samples = 0
                for data in val_loader:
                    data = data.to(DEVICE)
                    if args.model in {"GIN", "GINNA", "GIN+", "MF"}:
                        output = model(data.x, data.edge_index, data.batch)
                    else:
                        output = model(
                            data.x, data.edge_index, data.edge_attr, data.batch
                        )
                    if args.loss == "weighted_BCE":
                        class_weights = compute_class_weight(
                            class_weight="balanced",
                            classes=np.unique(data.y.cpu()),
                            y=np.array(data.y.cpu()),
                        )
                        batch_validation_loss = loss_function(
                            output[:, 0].to(float), data.y.to(float), class_weights
                        )
                    else:
                        batch_validation_loss = loss_function(
                            output[:, 0].to(float), data.y.to(float)
                        )
                    batch_validation_losses.append(batch_validation_loss.item())
                    num_validation_samples += len(data.batch)

            validation_losses.append(
                np.sum(np.array(batch_validation_losses)) / num_validation_samples
            )

            early_stopper(np.sum(np.array(batch_validation_losses)))
            if early_stopper.early_stop:
                break

            scheduler.step(np.sum(np.array(batch_validation_losses)))

        plot_losses(
            training_losses,
            validation_losses,
            os.path.join(args.outputDirectory, str(fold_idx_ext + 1) + "/loss.png"),
        )

        # Apply retrained model to test data
        model.eval()
        with torch.no_grad():
            targets = []
            preds_list = []
            for data in test_loader:
                data = data.to(DEVICE)
                if args.model in {"GIN", "GINNA", "GIN+", "MF"}:
                    output = model(data.x, data.edge_index, data.batch)
                else:
                    output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                targets.extend(data.y.tolist())
                preds_list.extend(output.tolist())

        # Compute test metrics of current fold with predictions on test set
        auc_pr = average_precision_score(
            np.array(targets), np.array(list(itertools.chain(*preds_list)))
        )
        rocauc = roc_auc_score(
            np.array(targets), np.array(list(itertools.chain(*preds_list)))
        )
        # Get best threshold
        if args.loss == "weighted_BCE":
            opt_threshold = 0.5
        else:
            fpr, tpr, thresholds = roc_curve(
                np.array(targets), np.array(list(itertools.chain(*preds_list)))
            )
            opt_threshold = thresholds[np.argmax(tpr - fpr)]
        # Compute binary predictions from probability predictions with best threshold
        preds_binary = np.array(list(itertools.chain(*preds_list))) > opt_threshold
        # Compute metrics that require binary predictions
        mcc = matthews_corrcoef(np.array(targets), preds_binary)
        precision = precision_score(np.array(targets), preds_binary)
        recall = recall_score(np.array(targets), preds_binary)

        opt_threshold_list.append(opt_threshold)
        mcc_list.append(mcc)
        precision_list.append(precision)
        recall_list.append(recall)
        auc_pr_list.append(auc_pr)
        rocauc_list.append(rocauc)

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(args.outputDirectory, str(fold_idx_ext + 1) + "/model.pt"),
        )

        # Save individual fold results
        with open(
            os.path.join(args.outputDirectory, "results_individual.csv"),
            "a",
            encoding="UTF8",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow([opt_threshold, mcc, precision, recall, auc_pr, rocauc])

        logging.info(
            f"End external CV fold {fold_idx_ext+1}/{args.numExternalCVFolds}."
        )

    #####################################################################
    #                           save results
    #####################################################################

    averages = [
        np.round(np.average(opt_threshold_list), 2),
        np.round(np.average(mcc_list), 2),
        np.round(np.average(precision_list), 2),
        np.round(np.average(recall_list), 2),
        np.round(np.average(auc_pr_list), 2),
        np.round(np.average(rocauc_list), 2),
    ]
    stddevs = [
        np.round(np.std(opt_threshold_list), 2),
        np.round(np.std(mcc_list), 2),
        np.round(np.std(precision_list), 2),
        np.round(np.std(recall_list), 2),
        np.round(np.std(auc_pr_list), 2),
        np.round(np.std(rocauc_list), 2),
    ]
    names = [
        "opt_threshold",
        "mcc",
        "precision",
        "recall",
        "auc_pr",
        "rocauc",
    ]
    with open(
        os.path.join(args.outputDirectory, "results_summary.csv"),
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "metric",
                "avg",
                "stdev",
            )
        )
        writer.writerows(zip(names, averages, stddevs))

    print("Done!")
