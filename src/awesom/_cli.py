import csv
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Annotated

import numpy as np
import optuna
import torch
import torch_geometric
import typer
import yaml
from rich.console import Console
from rich.json import JSON
from sklearn.model_selection import KFold
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from awesom import MetricsCalculator
from awesom.dataset import SOMDataset
from awesom.gpu_utils import get_optimal_batch_size
from awesom.metrics import THRESHOLD
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
                y_probs=predictions.get_probabilities().mean(dim=0),
                y_true=predictions.y_trues,
            )
        )

        # TODO: maybe write intermediate information

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
    rng = np.random.default_rng(seed)
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

    mol_id_to_smiles = {datum.mol_id[0].item(): datum.smiles for datum in data}

    with output_path.open("w", encoding="UTF-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "smiles",
                "mol_id",
                "atom_id",
                "y_true",
                "y_pred",
                "y_prob",
                "aleatoric_uncertainty",
                "epistemic_uncertainty",
                "total_uncertainty",
            ],
        )
        writer.writeheader()

        for mol_id, atom_id, true_label, probability, u_ale, u_epi, u_tot in zip(
            predictions.mol_ids.tolist(),
            predictions.atom_ids.tolist(),
            predictions.y_trues.tolist(),
            predictions.get_probabilities().mean(dim=0).tolist(),
            *predictions.get_uncertainties(),
        ):
            writer.writerow(
                {
                    "smiles": mol_id_to_smiles[mol_id],
                    "mol_id": mol_id,
                    "atom_id": atom_id,
                    "y_true": int(true_label),
                    "y_pred": int(probability < THRESHOLD),
                    "y_prob": np.round(probability, 2),
                    "aleatoric_uncertainty": np.round(u_ale.item(), 2),
                    "epistemic_uncertainty": np.round(u_epi.item(), 2),
                    "total_uncertainty": np.round(u_tot.item(), 2),
                }
            )


@app.command(
    name="metrics",
    help="Calculate metrics for existing SOM predictions.",
)
def metrics(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input prediction CSV file.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to output metrics JSON file.",
        ),
    ] = None,
    n_bootstrap_samples: Annotated[
        int | None,
        typer.Option(
            "--bootstrap",
            help="Number of bootstrapping samples to perform.",
        ),
    ] = None,
):
    def compute_metrics(y_true, y_prob, mol_ids):
        return MetricsCalculator.compute_torchmetrics(
            y_true=torch.from_numpy(y_true), y_probs=torch.from_numpy(y_prob)
        ) | {
            "top2_rate": MetricsCalculator.compute_top2_accuracy(
                y_true=torch.from_numpy(y_true),
                y_probs=torch.from_numpy(y_prob),
                mol_ids=torch.from_numpy(mol_ids),
            )
        }

    rows = [row for row in csv.DictReader(input_path.open())]

    smiles_full = np.array([row["smiles"] for row in rows], dtype=str)
    mol_ids_full = np.array([row["mol_id"] for row in rows], dtype=int)
    y_true_full = np.array([int(row["y_true"]) for row in rows], dtype=bool)
    y_pred_full = np.array([int(row["y_pred"]) for row in rows], dtype=bool)
    y_prob_full = np.array([row["y_prob"] for row in rows], dtype=float)

    computed_metrics_samples = []
    unique_smiles = np.unique(smiles_full)
    rng = np.random.default_rng(0)

    if n_bootstrap_samples:
        with stderr.status(f"Performing {n_bootstrap_samples} bootstraps"):
            for _ in range(n_bootstrap_samples):
                counter: Counter[str] = Counter(
                    rng.choice(unique_smiles, size=len(unique_smiles), replace=True)
                )
                repeats = [counter[it] for it in smiles_full]

                smiles = np.repeat(smiles_full, repeats)
                mol_ids = np.repeat(mol_ids_full, repeats)
                y_true = np.repeat(y_true_full, repeats)
                y_pred = np.repeat(y_pred_full, repeats)
                y_prob = np.repeat(y_prob_full, repeats)

                computed_metrics_samples.append(
                    compute_metrics(y_true, y_prob, mol_ids)
                )
    else:
        computed_metrics_samples.append(
            compute_metrics(y_true_full, y_prob_full, mol_ids_full)
        )

    computed_metrics = {
        key: {
            "mean": np.round(np.mean(values), 4),
            "std": np.round(np.std(values), 4),
        }
        for key, values in {
            key: [sample[key] for sample in computed_metrics_samples]
            for key in computed_metrics_samples[0].keys()
        }.items()
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(computed_metrics, indent=4))
    else:
        stderr.print(JSON.from_data(computed_metrics))


@app.command(
    name="hyperparameters",
    help="Perform CV hyperparameter search for an aweSOM model ensemble.",
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
    ] = 100,
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
            num_folds,
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
