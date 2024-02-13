import argparse
import optuna
import os
import torch

from lightning import Trainer, Callback
from lightning import seed_everything as lightning_seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from operator import itemgetter
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.core.saving import save_hparams_to_yaml
from sklearn.model_selection import KFold
from statistics import mean
from torch_geometric import transforms as T
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric.loader import DataLoader

from awesom import (
    SOM,
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
    M12,
    M13,
    M14,
    M15,
    ValidationMetrics,
)

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
    "M12": M12,
    "M13": M13,
    "M14": M14,
    "M15": M15,
}


class PatchedCallback(PyTorchLightningPruningCallback, Callback):
    pass


import argparse
import optuna
import os
import torch

from lightning import seed_everything as lightning_seed_everything
from lightning import Trainer, Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from operator import itemgetter
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from sklearn.model_selection import KFold
from torch_geometric import transforms as T
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric.loader import DataLoader

from awesom import (
    SOM,
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
    M12,
    M13,
    M14,
    M15,
    ValidationMetrics,
)

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
    "M12": M12,
    "M13": M13,
    "M14": M14,
    "M15": M15,
}


class PatchedCallback(PyTorchLightningPruningCallback, Callback):
    pass


def run_train():
    lightning_seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    data = SOM(root=args.inputFolder, transform=T.ToUndirected()).shuffle()

    data_params = dict(
        num_node_features=data.num_node_features,
        num_edge_features=data.num_edge_features,
    )

    def objective(trial):
        metric_lst = []
        kfold = KFold(n_splits=args.numCVFolds, shuffle=True, random_state=42)

        for fold_id, (train_idx, val_idx) in enumerate(kfold.split(range(len(data)))):
            train_data = itemgetter(*train_idx)(data)
            val_data = itemgetter(*val_idx)(data)

            train_loader = DataLoader(train_data, batch_size=args.batchSize)
            val_loader = DataLoader(val_data, batch_size=args.batchSize)

            print(
                f"CV-fold {fold_id}/{args.numCVFolds}: number of training instances {len(train_data)}, number of validation instances {len(val_data)}"
            )

            hyperparams = model_dict[args.model].get_params(trial)
            model = GNN(
                params=data_params,
                hyperparams=hyperparams,
                architecture=args.model,
                pos_weight=data.get_pos_weight(),
            )

            tbl = TensorBoardLogger(
                save_dir=os.path.join(args.outputFolder, "logs"),
                name=f"trial{trial._trial_id}",
                version=f"fold{fold_id}",
                default_hp_metric=False,
            )

            callbacks = [
                EarlyStopping(monitor="val/mcc", mode="max", min_delta=0, patience=30),
                PatchedCallback(trial=trial, monitor="val/mcc"),
                ModelCheckpoint(
                    filename=f"trial{trial._trial_id}", monitor="val/mcc", mode="max"
                ),
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

            metric_lst.append(trainer.callback_metrics["val/mcc"].item())

        return mean(metric_lst)

    if not os.path.exists(args.outputFolder):
        os.makedirs(args.outputFolder)

    storage = "sqlite:///" + args.outputFolder + "/storage.db"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.numberOptunaTrials, gc_after_trial=True)

    print(
        f"Best trial is trial {study.best_trial._trial_id} with mean validation MCC {study.best_trial.value} and hyperparameters:"
    )
    for key, value in study.best_trial.params.items():
        print("   {}: {}".format(key, value))

    save_hparams_to_yaml(
        os.path.join(args.outputFolder, "best_hparams.yaml"), study.best_trial.params
    )

    # Compute and log all relevant validation metrics from models trained with optimal hparams
    # for subsequent manual model analysis and selection
    collected_validation_outputs = {}
    kfold = KFold(n_splits=args.numCVFolds, shuffle=True, random_state=42)
    for fold_id, (_, val_idx) in enumerate(kfold.split(range(len(data)))):
        val_data = itemgetter(*val_idx)(data)
        val_loader = DataLoader(val_data, batch_size=args.batchSize)
        model = GNN(
            params=data_params,
            hyperparams=study.best_trial.params,
            architecture=args.model,
            pos_weight=data.get_pos_weight(),
        )

        checkpoint_path = os.path.join(
            os.path.join(
                os.path.join(
                    os.path.join(
                        os.path.join(args.outputFolder, "logs"),
                        f"trial{study.best_trial._trial_id}",
                    ),
                    f"fold{fold_id}",
                ),
                "checkpoints",
            ),
            f"trial{study.best_trial._trial_id}.ckpt",
        )

        model = GNN.load_from_checkpoint(checkpoint_path)

        trainer = Trainer(accelerator="auto", logger=False)
        collected_validation_outputs[fold_id] = trainer.predict(
            model=model, dataloaders=val_loader
        )

    ValidationMetrics.compute_and_log_validation_metrics(
        collected_validation_outputs, args.outputFolder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finding the optinal hyperparameters via Optuna k-fold-cross-validation.")

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
        "-b",
        "--batchSize",
        type=int,
        required=True,
        help="The batch size.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=True,
        help="The maximum number of training epochs.",
    )
    parser.add_argument(
        "-n",
        "--numCVFolds",
        type=int,
        required=True,
        help="The number of cross-validation folds.",
    )
    parser.add_argument(
        "-t",
        "--numberOptunaTrials",
        type=int,
        required=True,
        help="The number of Optuna trials.",
    )

    args = parser.parse_args()
    run_train()
