import os
from pathlib import Path
from typing import Annotated

import numpy as np
import torch_geometric
import typer
import yaml
from rich.console import Console
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.dataset import SOMDataset
from awesom.metrics import ResultsLogger
from awesom.model import SOMPredictor, predict_ensemble

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
        int,
        typer.Option(
            "--batch-size",
            help="Batch size during training.",
        ),
    ] = 32,
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
        train_loader: DataLoader = DataLoader(data, batch_size=batch_size, shuffle=True)

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
            help="Path to output prediction directory.",
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
    help="...",
)
def hyperparameters():
    raise NotImplementedError()
