import argparse
import optuna
import os
import torch

from lightning import Trainer, Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from operator import itemgetter
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from sklearn.model_selection import KFold
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
from awesom.utils import (
    seed_everything,
)

NFOLDS = 5
# BATCHSIZE = 32
THRESHOLD = 0.2


class PatchedCallback(PyTorchLightningPruningCallback, Callback):
    pass


def run_train():
    data = SOM(root=args.inputFolder).shuffle()  # , transform=T.Distance(norm=False)

    print(f"Number of training graphs: {len(data)}")

    fold = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    for fold_id, (train_idx, val_idx) in enumerate(fold.split(range(len(data)))):
        train_data = itemgetter(*train_idx)(data)
        val_data = itemgetter(*val_idx)(data)
        train_loader = DataLoader(train_data, batch_size=len(train_data))
        val_loader = DataLoader(val_data, batch_size=len(val_data))

        def objective(trial):
            if args.model == "GATv2":
                params, hyperparams = GATv2.get_params(data, trial)
                model = GATv2(params, hyperparams)
            elif args.model == "GIN":
                params, hyperparams = GIN.get_params(data, trial)
                model = GIN(params, hyperparams)
            elif args.model == "GINE":
                params, hyperparams = GINE.get_params(data, trial)
                model = GINE(params, hyperparams)
            elif args.model == "NN":
                params, hyperparams = NN.get_params(data, trial)
                model = NN(params, hyperparams)

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
            )

            trainer.fit(
                model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

            return trainer.callback_metrics["val/loss"].item()

        if not os.path.exists(os.path.join(args.logFolder, f"fold{fold_id}")):
            os.makedirs(os.path.join(args.logFolder, f"fold{fold_id}"))

        storage = "sqlite:///" + args.logFolder + f"/fold{fold_id}" + "/storage.db"
        pruner = optuna.pruners.MedianPruner(n_min_trials=5, n_warmup_steps=50)
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

        with open(os.path.join(args.logFolder, "best_model_paths.txt"), "a") as f:
            f.write(
                f"{args.logFolder}/fold{fold_id}/logs/trial{study.best_trial._trial_id}/version_0/checkpoints/\n"
            )


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

    seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    run_train()
