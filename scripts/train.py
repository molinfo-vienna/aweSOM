import argparse
import csv
import optuna
import os
import torch

from lightning import Trainer, Callback, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from operator import itemgetter
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from sklearn.model_selection import KFold
from statistics import mean, stdev
from torch_geometric import transforms as T
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric.loader import DataLoader
from torchmetrics import MatthewsCorrCoef, AUROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall

from awesom.dataset import SOM
from awesom.metrics_utils import compute_ranking
from awesom.models import (
    GATv2,
    GIN,
    GINE,
    GINED,
    MF,
    Cheb,
    GNN,
)

NFOLDS = 10
BATCHSIZE = 64

mcc = MatthewsCorrCoef(task="binary")
auroc = AUROC(task="multiclass", num_classes=2)
precision = BinaryPrecision(threshold=0.5)
recall = BinaryRecall(threshold=0.5)


class PatchedCallback(PyTorchLightningPruningCallback, Callback):
    pass


def run_train():
    seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    data = SOM(root=args.inputFolder).shuffle()  # , transform=T.Distance(norm=False)
    class_weights = data.get_class_weights()

    print(f"Number of training graphs: {len(data)}")

    model_dict = {
        "GATv2": GATv2,
        "GIN": GIN,
        "GINE": GINE,
        "GINED": GINED,
        "MF": MF,
        "Cheb": Cheb,
    }

    aurocs = []
    mccs = []
    precisions = []
    recalls = []
    mol_r_precisions = []
    mol_aurocs = []
    top2s = []
    atom_r_precisions = []

    fold = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    for fold_id, (train_idx, val_idx) in enumerate(fold.split(range(len(data)))):
        train_data = itemgetter(*train_idx)(data)
        val_data = itemgetter(*val_idx)(data)
        train_loader = DataLoader(train_data, batch_size=len(data))
        val_loader = DataLoader(val_data, batch_size=len(data))

        def objective(trial):
            model_type = model_dict[args.model]
            params, hyperparams = model_type.get_params(data, trial)
            model = GNN(params, hyperparams, class_weights, model_type)

            tbl = TensorBoardLogger(
                save_dir=os.path.join(
                    os.path.join(args.logFolder, f"fold{fold_id}"), "logs",
                ),
                name=f"trial{trial._trial_id}",
                default_hp_metric=False,
            )
            callbacks = [
                EarlyStopping(monitor="val/loss", mode="min", min_delta=0, patience=30),
                PatchedCallback(trial=trial, monitor="val/loss"),
                ModelCheckpoint(filename=f"trial{trial._trial_id}", monitor="val/loss", mode="min")
            ]

            trainer = Trainer(
                accelerator="auto",
                max_epochs=args.epochs,
                logger=tbl,
                log_every_n_steps=1,
                callbacks=callbacks,
                precision="16-mixed",
            )

            trainer.fit(
                model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

            return trainer.callback_metrics["val/loss"].item()

        if not os.path.exists(os.path.join(args.logFolder, f"fold{fold_id}")):
            os.makedirs(os.path.join(args.logFolder, f"fold{fold_id}"))

        storage = "sqlite:///" + args.logFolder + f"/fold{fold_id}" + "/storage.db"
        pruner = optuna.pruners.MedianPruner(n_min_trials=5, n_warmup_steps=5000)
        study = optuna.create_study(
            study_name=f"{args.model}_study",
            direction="minimize",
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=args.numberTrials, gc_after_trial=True)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("   Number of finished trials: ", len(study.trials))
        print("   Number of pruned trials: ", len(pruned_trials))
        print("   Number of complete trials: ", len(complete_trials))
        print(
            f"Best trial is trial {study.best_trial._trial_id} with validation loss {study.best_trial.value} and hyperparameters:"
        )
        for key, value in study.best_trial.params.items():
            print("   {}: {}".format(key, value))

        path = f"{args.logFolder}/fold{fold_id}/logs/trial{study.best_trial._trial_id}/version_0/checkpoints/"
        with open(os.path.join(args.logFolder, "best_model_paths.txt"), "a") as f:
            f.write(path + "\n")

        ##### Recompute validation metrics and log them into validation.txt #####

        # Load best models
        for file in os.listdir(path):
            if file.endswith(".ckpt"):
                path = os.path.join(path, file)
        model = GNN.load_from_checkpoint(path)

        # Make predictions on validation split with best model
        trainer = Trainer(accelerator="auto", logger=False)
        out = trainer.predict(
            model=model, dataloaders=DataLoader(val_data, batch_size=len(val_data))
        )
        y_hat = out[0][0]
        y = out[0][1]
        mol_id = out[0][2]
        atom_id = out[0][3]

        y_hat_bin = torch.max(y_hat, dim=1).indices

        # Compute normal metrics
        mccs.append(mcc(y_hat_bin, y).item())
        precisions.append(precision(y_hat_bin, y).item())
        recalls.append(recall(y_hat_bin, y).item())
        aurocs.append(auroc(y_hat, y).item())

        # Compute the atom ranking for the current validation fold
        # and write the detailed, per atom results into validation_fold{fold_id}.csv
        ranking = compute_ranking(y_hat, mol_id)
        with open(os.path.join(args.logFolder, f"validation_fold{fold_id}.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    "probabilities",
                    "ranking",
                    "predicted_labels",
                    "true_labels",
                    "mol_id",
                    "atom_id",
                )
            )
            for row in zip(
                y_hat[:, 1].tolist(),
                ranking.tolist(),
                y_hat_bin.tolist(),
                y.tolist(),
                mol_id.tolist(),
                atom_id.tolist(),
            ):
                writer.writerow(row)
    
        # Compute weird metrics from Porokhin's GNN-SOM paper for comparison's sake...
        ranked_y_hat_bin = torch.index_select(y_hat_bin, 0, ranking)
        total_num_soms_in_validation_split = torch.sum(y).item()
        atom_r_precisions.append(torch.sum(ranked_y_hat_bin[:total_num_soms_in_validation_split]).item() / total_num_soms_in_validation_split)

        top2_correctness_rate = 0
        per_molecule_aurocs = []
        per_molecule_r_precisions = []
        for mid in list(dict.fromkeys(mol_id.tolist())):  # This is a somewhat complicated way to get an ordered set, but it works
            mask = torch.where(mol_id == mid)[0]
            masked_ranking = ranking[mask]
            masked_y = y[mask]
            masked_y_hat = y_hat[mask]
            masked_y_hat_bin = y_hat_bin[mask]
            masked_ranked_y_hat_bin = torch.index_select(masked_y_hat_bin, 0, masked_ranking)
            num_soms_in_current_mol = torch.sum(masked_y).item()
            top2_correctness_rate += 1 if torch.sum(masked_ranked_y_hat_bin[:2]).item() > 0 else 0
            per_molecule_aurocs.append(auroc(masked_y_hat, masked_y).item())
            per_molecule_r_precisions.append(torch.sum(masked_ranked_y_hat_bin[:num_soms_in_current_mol]).item() / num_soms_in_current_mol)
        top2_correctness_rate /= len(set(mol_id.tolist()))
        top2s.append(top2_correctness_rate)
        mol_aurocs.append(mean(per_molecule_aurocs))
        mol_r_precisions.append(mean(per_molecule_r_precisions))
    
    # Write metrics to a text file
    with open(os.path.join(args.logFolder, "validation.txt"), "w") as f:
        f.write(f"MCC: {round(mean(mccs), 2)} +/- {round(stdev(mccs), 2)}\n")
        f.write(f"Precision: {round(mean(precisions), 2)} +/- {round(stdev(precisions), 2)}\n")
        f.write(f"Recall: {round(mean(recalls), 2)} +/- {round(stdev(recalls), 2)}\n")
        f.write(f"Molecular R-Precision: {round(mean(mol_r_precisions), 2)} +/- {round(stdev(mol_r_precisions), 2)}\n")
        f.write(f"Molecular AUROC:  {round(mean(mol_aurocs), 2)} +/- {round(stdev(mol_aurocs), 2)}\n")
        f.write(f"Top-2 Correctness Rate: {round(mean(top2s), 2)} +/- {round(stdev(top2s), 2)}\n")
        f.write(f"Atomic R-Precision: {round(mean(atom_r_precisions), 2)} +/- {round(stdev(atom_r_precisions), 2)}\n")
        f.write(f"Atomic AUROC: {round(mean(aurocs), 2)} +/- {round(stdev(aurocs), 2)}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training and testing the model.")

    parser.add_argument(
        "-i",
        "--inputFolder",
        type=str,
        required=True,
        help="The folder where the input data is stored.",
    )
    parser.add_argument(
        "-l",
        "--logFolder",
        type=str,
        required=True,
        help="The name of the log folder.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The desired model architecture.",
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
        "--numberTrials",
        type=int,
        required=True,
        help="The number of Optuna trials.",
    )

    args = parser.parse_args()
    run_train()
