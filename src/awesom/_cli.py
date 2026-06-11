import os
import statistics
from pathlib import Path
from typing import Annotated

import numpy as np
import optuna
import torch
import torch_geometric
import typer
import yaml
from rich.console import Console
from sklearn.model_selection import KFold
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from awesom import MetricsCalculator
from awesom.dataset import SOMDataset
from awesom.gpu_utils import get_device
from awesom.metrics import ResultsLogger
from awesom.model import GINEWithContextPooling, SOMPredictor, predict_ensemble

app = typer.Typer(add_completion=False)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


stdout = Console()
stderr = Console(stderr=True)


def load_models(checkpoints_path: Path) -> list[SOMPredictor]:
    models = []

    for model_path in sorted(checkpoints_path.glob("model_*")):
        checkpoint_path = model_path / "checkpoints" / "best_model.ckpt"
        if checkpoint_path.exists():
            models.append(SOMPredictor.load(str(checkpoint_path)))

    if not models:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoints_path}")

    return models


def get_optimal_batch_size() -> int:
    """Automatically determine optimal batch size based on GPU memory."""
    if torch.cuda.is_available():
        device = get_device()
        gpu_memory = (
            torch.cuda.get_device_properties(device).total_memory / 1024**3
        )  # GB

        if gpu_memory >= 24:  # 24GB+ GPU (e.g., RTX 4090, A100)
            return 256
        elif gpu_memory >= 16:  # 16-24GB GPU (e.g., RTX 4080, V100)
            return 192
        elif gpu_memory >= 12:  # 12-16GB GPU (e.g., RTX 3080 Ti)
            return 128
        elif gpu_memory >= 8:  # 8-12GB GPU (e.g., RTX 3080, RTX 4070)
            return 96
        else:  # <8GB GPU
            return 64
    else:
        return 32  # CPU fallback


def objective(
    trial: optuna.trial.Trial,
    data: Dataset,
    num_folds: int,
    max_epochs: int,
    batch_size: int,
    output_path: Path,
) -> float:
    def compute_and_save_average_metrics(
        metrics_list: list[dict[str, float]],
    ) -> dict[str, float]:
        avg_metrics = {}
        metric_names: set[str] = set.intersection(
            *(set(metrics.keys()) for metrics in metrics_list)
        )

        with (output_path / "validation.txt").open("w") as f:
            for metric_name in metric_names:
                values = [metrics[metric_name] for metrics in metrics_list]
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                avg_metrics[metric_name] = mean_val
                f.write(
                    f"{metric_name}: {round(mean_val, 4)} +/- {round(std_val, 4)}\n"
                )

        return avg_metrics

    data_params = {
        "num_node_features": data.num_node_features,
        "num_edge_features": data.num_edge_features,
    }

    hyperparams: dict[str, int | float] = GINEWithContextPooling.get_params(trial)
    hyperparams["epochs"] = max_epochs

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_metrics = []
    fold_epochs = []

    for fold, (train_idx, val_idx) in enumerate(
        tqdm(kfold.split(data), total=num_folds, desc=f"Trial {trial.number}")
    ):
        print(f"Trial {trial.number}, Fold {fold + 1}/{num_folds}")

        train_data = data[train_idx]
        val_data = data[val_idx]

        assert isinstance(train_data, Dataset)
        assert isinstance(val_data, Dataset)

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between epochs
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between epochs
        )

        model = SOMPredictor(data_params, hyperparams)
        actual_epochs = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            patience=20,
        )

        fold_epochs.append(actual_epochs)
        print(f"  Fold {fold + 1} stopped at epoch {actual_epochs}")

        # Evaluate on validation set and collect predictions
        model.eval()

        predictions = predict_ensemble(val_loader, [model])

        fold_metrics.append(
            MetricsCalculator.compute_torchmetrics(
                y_probs=predictions.get_probabilities()[0],
                y_true=predictions.y_trues,
            )
        )

        ResultsLogger(str(output_path / f"fold_{fold}")).save_results(
            predictions, mode="test"
        )

    optimal_epochs = int(sum(fold_epochs) / len(fold_epochs))
    trial.set_user_attr("optimal_epochs", optimal_epochs)

    metrics = compute_and_save_average_metrics(fold_metrics)

    return metrics["MCC"]


@app.command(
    name="train",
    help="Train an aweSOM model ensemble on SOM data.",
)
def train(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input training data (SDF, SMILES).",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to model output directory.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to config file containing hyperparameters (YAML).",
        ),
    ],
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Global seed used for any sources of randomness.",
        ),
    ] = 42,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            help="Batch size during training.",
        ),
    ] = None,
    ensemble_size: Annotated[
        int,
        typer.Option(
            "--ensemble-size",
            help="Number of models to train for ensemble.",
        ),
    ] = 10,
) -> None:
    rng = np.random.default_rng(0)
    seeds = rng.choice(1000, ensemble_size, replace=False)

    data = SOMDataset(root=str(input_path), transform=T.ToUndirected())
    data_params = {
        "num_node_features": data.num_node_features,
        "num_edge_features": data.num_edge_features,
    }

    hyperparams = yaml.safe_load(config_path.open("r"))

    for i, seed in enumerate(seeds):
        stderr.print(f"Training model {i + 1}/{ensemble_size} with seed {seed}...")

        torch_geometric.seed_everything(seed)

        model = SOMPredictor(data_params, hyperparams)
        train_loader: DataLoader = DataLoader(
            data, batch_size=batch_size or get_optimal_batch_size(), shuffle=True
        )

        model.fit(
            train_loader=train_loader,
            max_epochs=int(hyperparams["epochs"]),
            log_dir=None,
            checkpoint_dir=str(output_path / f"model_{i}" / "checkpoints"),
            patience=20,
        )


@app.command(
    name="predict",
    help="Predict SOMs using an existing aweSOM model ensemble.",
)
def predict(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input data for which to predict SOMs (SDF, SMILES).",
        ),
    ],
    models_path: Annotated[
        Path,
        typer.Option(
            "-m",
            "--models",
            help="Path to directory containing aweSOM model ensemble.",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to prediction output directory.",
        ),
    ],
):
    data = SOMDataset(root=str(input_path), labeled=True, transform=T.ToUndirected())
    dataloader: DataLoader = DataLoader(data, batch_size=len(data), shuffle=False)

    models = load_models(models_path)
    predictions = predict_ensemble(dataloader, models)

    ResultsLogger(str(output_path)).save_results(predictions, mode="inference")


@app.command(
    name="metrics",
    help="Calculate metrics for existing SOM predictions.",
)
def metrics():
    raise NotImplementedError()


@app.command(
    name="hyperparameters",
    help="Perform CV hyperparameter search for a aweSOM model ensemble.",
)
def hyperparameters(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input training data (SDF, SMILES).",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output directory.",
        ),
    ],
    num_epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            help="Maximum number of epochs to train for.",
        ),
    ] = 500,
    num_folds: Annotated[
        int,
        typer.Option(
            "--folds",
            help="Number of CV folds to use during cross validation.",
        ),
    ] = 10,
    num_trials: Annotated[
        int,
        typer.Option(
            "--trials",
            help="Number of trials to run for hyperparameter search.",
        ),
    ] = 20,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            help="Batch size during training (default).",
        ),
    ] = None,
):
    output_path.mkdir(exist_ok=True, parents=True)

    study = optuna.create_study(
        direction="maximize",
        load_if_exists=True,
        storage=f"sqlite:///{output_path}/study.db",
        study_name="cv_hp_search",
    )

    data = SOMDataset(root=str(input_path), transform=T.ToUndirected()).shuffle()
    assert isinstance(data, Dataset)

    study.optimize(
        lambda trial: objective(
            trial,
            data,
            num_trials,
            num_epochs,
            batch_size or get_optimal_batch_size(),
            output_path,
        ),
        n_trials=num_trials,
    )

    best_params = study.best_trial.params
    best_params["epochs"] = study.best_trial.user_attrs["optimal_epochs"]

    with (output_path / "best_hparams.yaml").open("w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
