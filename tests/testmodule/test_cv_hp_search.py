import os
import warnings
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

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

INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "train")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "cv_hp_search")
HPARAMS_YAML_PATH = os.path.join(os.path.dirname(__file__))
MODEL = "M7"
EPOCHS = 10
NUM_CV_FOLDS = 2
NUM_OPTUNA_TRIALS = 2

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


def test_cv_hp_search() -> None:

    data = SOM(root=INPUT_PATH, transform=T.ToUndirected()).shuffle()
    data_params = dict(
        num_node_features=data.num_node_features,
        num_edge_features=data.num_edge_features,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial.set_user_attr("architecture", MODEL)

        performance_per_fold = []
        num_epochs_per_fold = []

        kfold = KFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=42)

        for fold_id, (train_idx, val_idx) in enumerate(kfold.split(range(len(data)))):
            train_data = itemgetter(*train_idx)(data)
            val_data = itemgetter(*val_idx)(data)

            print(
                f"CV-fold {fold_id}/{NUM_CV_FOLDS}: \
                number of training instances {len(train_data)}, number of validation instances {len(val_data)}"
            )

            hyperparams = MODELS[MODEL].get_params(trial)  # type: ignore[attr-defined]
            model = GNN(
                params=data_params,
                hyperparams=hyperparams,
                architecture=MODEL,
            )

            train_loader, val_loader = prepare_data_loaders(train_data, val_data)

            tbl = TensorBoardLogger(
                save_dir=Path(OUTPUT_PATH, "logs"),
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
                max_epochs=EPOCHS,
                logger=tbl,
                log_every_n_steps=1,
                callbacks=callbacks,
            )

            trainer.fit(
                model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

            performance_per_fold.append(trainer.callback_metrics["val/mcc"].item())
            num_epochs_per_fold.append(trainer.current_epoch + 1)

        avg_performance = mean(performance_per_fold)
        avg_num_epochs = int(mean(num_epochs_per_fold))

        trial.set_user_attr("epochs", avg_num_epochs)

        return avg_performance

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    storage = "sqlite:///" + OUTPUT_PATH + "/storage.db"
    study = optuna.create_study(
        storage=storage,
        study_name="optuna_study",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=NUM_OPTUNA_TRIALS, gc_after_trial=True)

    best_trial = study.best_trial

    print(
        f"Best trial is trial {best_trial._trial_id} with mean validation MCC {best_trial.value} and hyperparameters:"
    )
    for key, value in best_trial.params.items():
        print("   {}: {}".format(key, value))

    best_trial.params["epochs"] = best_trial.user_attrs.get("epochs", "N/A")
    best_trial.params["architecture"] = best_trial.user_attrs.get("architecture", "N/A")

    save_hparams_to_yaml(Path(OUTPUT_PATH, "best_hparams.yaml"), best_trial.params)

    # Compute and log all relevant validation metrics from models trained with optimal hparams
    # for subsequent manual model analysis and selection
    collected_validation_outputs: dict[int, List[torch.Tensor]] = {}
    kfold = KFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=42)
    for fold_id, (_, val_idx) in enumerate(kfold.split(range(len(data)))):
        val_data = itemgetter(*val_idx)(data)
        val_loader = DataLoader(
            val_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        hyperparams = study.best_trial.params
        model = GNN(
            params=data_params,
            hyperparams=hyperparams,
            architecture=MODEL,
        )

        checkpoint_path = Path(
            Path(
                Path(
                    Path(
                        Path(OUTPUT_PATH, "logs"),
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
        predictions = trainer.predict(model=model, dataloaders=val_loader)

        if predictions is not None and len(predictions) > 0:
            predictions = predictions[0]
        else:
            raise ValueError("No predictions were made.")

        collected_validation_outputs[fold_id] = predictions[:-1]  # type: ignore[assignment]
        descriptions = predictions[-1]

    ValidationLogger.compute_and_log_validation_results(
        collected_validation_outputs, descriptions, OUTPUT_PATH
    )
