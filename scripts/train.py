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
from pytorch_lightning.core.saving import save_hparams_to_yaml
from sklearn.model_selection import KFold
from statistics import mean, stdev
from torch_geometric import transforms as T
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric.loader import DataLoader
from torchmetrics import MatthewsCorrCoef, AUROC, ROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall

from awesom.dataset import SOM
from awesom.metrics_utils import compute_ranking
from awesom.models import (
    GNN,
    M1,
    M2,
    M3,
    M4,
    M5,
    M6,
    M7,
    M8,
    M9,
    M10,
    M11,
)

NFOLDS = 10
BATCHSIZE = 64

mcc = MatthewsCorrCoef(task="binary")
auroc = AUROC(task="binary")
precision = BinaryPrecision()
recall = BinaryRecall()
roc = ROC(task="binary")

model_dict = {
    "M1": M1,
    "M2": M2,
    "M3": M3,
    "M4": M4,
    "M5": M5,
    "M6": M6,
    "M7": M7,
    "M8": M8,
    "M9": M9,
    "M10": M10,
    "M11": M11,
}

class PatchedCallback(PyTorchLightningPruningCallback, Callback):
    pass


def run_train():
    seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    data = SOM(root=args.inputFolder, transform=T.ToUndirected()).shuffle()  # , transform=T.Distance(norm=False)
    print(f"Number of training graphs: {len(data)}")

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
            model = GNN(params=params, 
                        hyperparams=hyperparams, 
                        class_weights=data.get_class_weights(), 
                        architecture=args.model, 
                        threshold=0.5
            )

            tbl = TensorBoardLogger(
                save_dir=os.path.join(
                    os.path.join(args.outputFolder, f"fold{fold_id}"), "logs",
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
            )

            trainer.fit(
                model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

            return trainer.callback_metrics["val/loss"].item()

        if not os.path.exists(os.path.join(args.outputFolder, f"fold{fold_id}")):
            os.makedirs(os.path.join(args.outputFolder, f"fold{fold_id}"))

        storage = "sqlite:///" + args.outputFolder + f"/fold{fold_id}" + "/storage.db"
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

        path = f"{args.outputFolder}/fold{fold_id}/logs/trial{study.best_trial._trial_id}/version_0/"
        with open(os.path.join(args.outputFolder, "best_model_paths.txt"), "a") as f:
            f.write(path + "\n")

        ##### Recompute validation metrics and log them into validation.txt #####

        # Load best models
        for file in os.listdir(os.path.join(path, "checkpoints")):
            if file.endswith(".ckpt"):
                checkpoint_path = os.path.join(os.path.join(path, "checkpoints"), file)
        for file in os.listdir(path):
            if file.endswith(".yaml"):
                hparams_file = os.path.join(path, file)

        model = GNN.load_from_checkpoint(checkpoint_path)

        # Make predictions on validation split with best model
        trainer = Trainer(accelerator="auto", logger=False)
        out = trainer.predict(
            model=model, dataloaders=DataLoader(val_data, batch_size=len(val_data))
        )
        y_hat = out[0][0]
        y = out[0][1]
        mol_id = out[0][2]
        atom_id = out[0][3]

        # Use this to compute binary predictions when using class weights
        y_hat_bin = torch.max(y_hat, dim=1).indices

        # Use this to compute binary predictions when using threshold moving
        # fpr, tpr, thresholds = roc(y_hat[:, 1], y)
        # best_threshold = thresholds[torch.argmax(tpr-fpr)]
        # y_hat_bin = (y_hat[:, 1] > best_threshold).to(int)
        # Overwrite threshold in hparams.yaml to be able to load it during inference
        # model.hparams.__setitem__("threshold", best_threshold.item())
        # save_hparams_to_yaml(hparams_file, model.hparams)

        # Compute the atom ranking for the current validation fold
        # and write the detailed, per atom results into validation_fold{fold_id}.csv
        ranking = compute_ranking(y_hat, mol_id)
        with open(os.path.join(args.outputFolder, f"validation_fold{fold_id}.csv"), "w") as f:
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

        # Compute normal metrics
        mccs.append(mcc(y_hat_bin, y).item())
        precisions.append(precision(y_hat_bin, y).item())
        recalls.append(recall(y_hat_bin, y).item())
        aurocs.append(auroc(y_hat[:, 1], y).item())
    
        # Compute weird metrics from Porokhin's GNN-SOM paper for comparison's sake...
        sorted_y = torch.index_select(y, dim=0, index=torch.sort(y_hat[:, 1], descending=True)[1])
        total_num_soms_in_validation_split = torch.sum(y).item()
        atom_r_precisions.append(torch.sum(sorted_y[:total_num_soms_in_validation_split]).item() / total_num_soms_in_validation_split)

        top2_correctness_rate = 0
        per_molecule_aurocs = []
        per_molecule_r_precisions = []
        for id in list(dict.fromkeys(mol_id.tolist())):  # This is a somewhat complicated way to get an ordered set, but it works
            mask = torch.where(mol_id == id)[0]
            masked_y = y[mask]
            masked_y_hat = y_hat[mask]
            masked_sorted_y = torch.index_select(masked_y, dim=0, index=torch.sort(masked_y_hat[:, 1], descending=True)[1])
            num_soms_in_current_mol = torch.sum(masked_y).item()
            if torch.sum(masked_sorted_y[:2]).item() > 0:
                top2_correctness_rate += 1 
            per_molecule_aurocs.append(auroc(masked_y_hat[:,1], masked_y).item())
            per_molecule_r_precisions.append(torch.sum(masked_sorted_y[:num_soms_in_current_mol]).item() / num_soms_in_current_mol)
        top2_correctness_rate /= len(set(mol_id.tolist()))
        top2s.append(top2_correctness_rate)
        mol_aurocs.append(mean(per_molecule_aurocs))
        mol_r_precisions.append(mean(per_molecule_r_precisions))
    
    # Write metrics to a text file
    with open(os.path.join(args.outputFolder, "validation.txt"), "w") as f:
        f.write(f"MCC: {round(mean(mccs), 2)} +/- {round(stdev(mccs), 2)}\n")
        f.write(f"Precision: {round(mean(precisions), 2)} +/- {round(stdev(precisions), 2)}\n")
        f.write(f"Recall: {round(mean(recalls), 2)} +/- {round(stdev(recalls), 2)}\n")
        f.write(f"Molecular R-Precision: {round(mean(mol_r_precisions), 2)} +/- {round(stdev(mol_r_precisions), 2)}\n")
        f.write(f"Molecular AUROC:  {round(mean(mol_aurocs), 2)} +/- {round(stdev(mol_aurocs), 2)}\n")
        f.write(f"Top-2 Correctness Rate: {round(mean(top2s), 2)} +/- {round(stdev(top2s), 2)}\n")
        f.write(f"Atomic R-Precision: {round(mean(atom_r_precisions), 2)} +/- {round(stdev(atom_r_precisions), 2)}\n")
        f.write(f"Atomic AUROC: {round(mean(aurocs), 2)} +/- {round(stdev(aurocs), 2)}\n")
    
    return None

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
        "-o",
        "--outputFolder",
        type=str,
        required=True,
        help="The name of the folder where the model's checkpoints and the validation metrics will be stored.",
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
