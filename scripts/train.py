import argparse
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
from torch_geometric.loader import DataLoader
from torchmetrics import MatthewsCorrCoef, AUROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall

from awesom.models import (
    GATv2,
    GIN,
    GINE,
    NN,
)
from awesom.dataset import SOM

NFOLDS = 10
BATCHSIZE = 64

mcc = MatthewsCorrCoef(task="binary")
auroc = AUROC(task="multiclass", num_classes=2)
precision = BinaryPrecision(threshold=0.5)
recall = BinaryRecall(threshold=0.5)


class PatchedCallback(PyTorchLightningPruningCallback, Callback):
    pass


def run_train():
    seed_everything(42, workers=True)

    data = SOM(root=args.inputFolder).shuffle()  # , transform=T.Distance(norm=False)
    class_weights = data.get_class_weights()

    print(f"Number of training graphs: {len(data)}")
    
    aurocs = []
    mccs = []
    precisions = []
    recalls = []

    fold = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    for fold_id, (train_idx, val_idx) in enumerate(fold.split(range(len(data)))):
        train_data = itemgetter(*train_idx)(data)
        val_data = itemgetter(*val_idx)(data)
        train_loader = DataLoader(train_data, batch_size=BATCHSIZE)
        val_loader = DataLoader(val_data, batch_size=BATCHSIZE)

        def objective(trial):
            if args.model == "GATv2":
                params, hyperparams = GATv2.get_params(data, trial)
                model = GATv2(params, hyperparams, class_weights)
            elif args.model == "GIN":
                params, hyperparams = GIN.get_params(data, trial)
                model = GIN(params, hyperparams, class_weights)
            elif args.model == "GINE":
                params, hyperparams = GINE.get_params(data, trial)
                model = GINE(params, hyperparams, class_weights)
            elif args.model == "NN":
                params, hyperparams = NN.get_params(data, trial)
                model = NN(params, hyperparams, class_weights)

            tbl = TensorBoardLogger(
                save_dir=os.path.join(
                    os.path.join(args.logFolder, f"fold{fold_id}"), "logs"
                ),
                name=f"trial{trial._trial_id}",
                default_hp_metric=False,
            )
            callbacks = []
            callbacks.append(
                EarlyStopping(monitor="val/loss", mode="min", min_delta=0, patience=10)
            )
            callbacks.append(PatchedCallback(trial=trial, monitor="val/loss"))
            callbacks.append(ModelCheckpoint(filename=f"trial{trial._trial_id}"))

            trainer = Trainer(
                accelerator="auto",
                max_epochs=args.epochs,
                logger=tbl,
                log_every_n_steps=1,
                callbacks=callbacks,
                deterministic=True,
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

        # Recompute validation metrics and log them into validation.txt

        for file in os.listdir(path):
            if file.endswith(".ckpt"):
                path = os.path.join(path, file)

        if args.model == "GATv2":
            model = GATv2.load_from_checkpoint(path)
        elif args.model == "GIN":
            model = GIN.load_from_checkpoint(path)
        elif args.model == "GINE":
            model = GINE.load_from_checkpoint(path)
        elif args.model == "NN":
            model = NN.load_from_checkpoint(path)

        trainer = Trainer(accelerator="auto", logger=False)
        out = trainer.predict(
            model=model, dataloaders=DataLoader(data, batch_size=len(data))
        )
        y_hat = out[0][0]
        y_hat_bin = torch.max(out[0][0], dim=1).indices
        y = out[0][1]
        aurocs.append(auroc(y_hat, y).item())
        mccs.append(mcc(y_hat_bin, y).item())
        precisions.append(precision(y_hat_bin, y).item())
        recalls.append(recall(y_hat_bin, y).item())

    with open(os.path.join(args.logFolder, "validation.txt"), "w") as f:
        f.write(f"AUROC: {round(mean(aurocs), 2)} +/- {round(stdev(aurocs), 2)}\n")
        f.write(f"MCC: {round(mean(mccs), 2)} +/- {round(stdev(mccs), 2)}\n")
        f.write(f"Precision: {round(mean(precisions), 2)} +/- {round(stdev(precisions), 2)}\n")
        f.write(f"Recall: {round(mean(recalls), 2)} +/- {round(stdev(recalls), 2)}\n")


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
        help="The desired model architecture. Choose between 'GATv2', 'GIN' and 'GINE'.",
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
