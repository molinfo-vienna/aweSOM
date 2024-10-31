import argparse
import os
import warnings
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import Tuple

import optuna
import torch
from lightning import Trainer
from lightning import seed_everything as lightning_seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.core.saving import save_hparams_to_yaml
from sklearn.model_selection import KFold
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.lightning_module import GNN
from awesom.metrics_utils import ValidationLogger
from awesom.models import M1, M2, M3, M4, M7, M9, M11, M12

warnings.filterwarnings("ignore", category=UserWarning)


MODELS = {
    "M1": M1,
    "M2": M2,
    "M3": M3,
    "M4": M4,
    "M7": M7,
    "M9": M9,
    "M11": M11,
    "M12": M12,
}

BATCH_SIZE = 32


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    lightning_seed_everything(seed)
    geometric_seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def prepare_data_loaders(
    train_data: Dataset, val_data: Dataset
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def main():
    set_seeds()

    data = SOM(root=args.inputPath, transform=T.ToUndirected()).shuffle()

    data_params = dict(
        num_node_features=data.num_node_features,
        num_edge_features=data.num_edge_features,
        # num_mol_features=data.mol_x.shape[1],
    )

    def objective(trial):
        metric_lst = []
        kfold = KFold(n_splits=args.numCVFolds, shuffle=True, random_state=42)

        for fold_id, (train_idx, val_idx) in enumerate(kfold.split(range(len(data)))):
            train_data = itemgetter(*train_idx)(data)
            val_data = itemgetter(*val_idx)(data)

            print(
                f"CV-fold {fold_id}/{args.numCVFolds}: number of training instances {len(train_data)}, number of validation instances {len(val_data)}"
            )

            hyperparams = MODELS[args.model].get_params(trial)
            hyperparams["mode"] = "cvhpsearch"
            model = GNN(
                params=data_params,
                hyperparams=hyperparams,
                architecture=args.model,
            )

            train_loader, val_loader = prepare_data_loaders(train_data, val_data)

            tbl = TensorBoardLogger(
                save_dir=Path(args.outputPath, "logs"),
                name=f"trial{trial._trial_id}",
                version=f"fold{fold_id}",
                default_hp_metric=False,
            )

            callbacks = [
                EarlyStopping(monitor="val/loss", mode="min", min_delta=0, patience=20),
                ModelCheckpoint(
                    filename=f"trial{trial._trial_id}", monitor="val/loss", mode="min"
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

    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)

    storage = "sqlite:///" + args.outputPath + "/storage.db"
    study = optuna.create_study(
        storage=storage,
        study_name="optuna_study",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=args.numberOptunaTrials, gc_after_trial=True)

    print(
        f"Best trial is trial {study.best_trial._trial_id} with mean validation MCC {study.best_trial.value} and hyperparameters:"
    )
    for key, value in study.best_trial.params.items():
        print("   {}: {}".format(key, value))

    save_hparams_to_yaml(
        Path(args.outputPath, "best_hparams.yaml"), study.best_trial.params
    )

    # Compute and log all relevant validation metrics from models trained with optimal hparams
    # for subsequent manual model analysis and selection
    collected_validation_outputs = {}
    kfold = KFold(n_splits=args.numCVFolds, shuffle=True, random_state=42)
    for fold_id, (_, val_idx) in enumerate(kfold.split(range(len(data)))):
        val_data = itemgetter(*val_idx)(data)
        val_loader = DataLoader(
            val_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=12,
            persistent_workers=True,
        )
        hyperparams = study.best_trial.params
        hyperparams["mode"] = "cvhpsearch"
        model = GNN(
            params=data_params,
            hyperparams=hyperparams,
            architecture=args.model,
        )

        checkpoint_path = Path(
            Path(
                Path(
                    Path(
                        Path(args.outputPath, "logs"),
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

    ValidationLogger.compute_and_log_validation_results(
        collected_validation_outputs, args.outputPath
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Finding the optinal hyperparameters via Optuna k-fold-cross-validation."
    )

    parser.add_argument(
        "-i",
        "--inputPath",
        type=str,
        required=True,
        help="Folder holding the input data.",
    )
    parser.add_argument(
        "-o",
        "--outputPath",
        type=str,
        required=True,
        help="Folder to which the output will be written. \
            The best hyperparameters will be stored in a YAML file. \
                The individual validation metrics of each fold will be stored in a CSV file. \
                    The best model's checkpoints will be stored in a directory. \
                        The averaged predictions made with the best hyperparameters will be stored in a text file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        default="M7",
        help="Model architecture.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=True,
        default=1000,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "-n",
        "--numCVFolds",
        type=int,
        required=True,
        default=10,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "-t",
        "--numberOptunaTrials",
        type=int,
        required=True,
        default=20,
        help="Number of optuna trials.",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    main()
    print("Finished in:")
    print(datetime.now() - start_time)
