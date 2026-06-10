from pathlib import Path
from typing import Annotated
import os

import numpy as np
import typer
from torch_geometric import transforms as T

from awesom import SOMDataset

app = typer.Typer(add_completion=False)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
    models_path: Annotated[
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
            default=42,
        ),
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size during training.",
            default=32,
        ),
    ],
    ensemble_size: Annotated[
        int,
        typer.Option(
            "--ensemble-size",
            help="Number of models to train for the ensemble.",
            default=10,
        ),
    ],
) -> None:
    rng = np.random.default_rng(0)
    seeds = rng.choice(1000, ensemble_size, replace=False)

    data = SOMDataset(root=str(input_path), transform=T.ToUndirected())

    for i, seed in enumerate(seeds):
        pass

    raise NotImplementedError()

@app.command(
    name="predict",
    help="Predict SOMs using an existing aweSOM model ensemble.",
)
def predict():
    raise NotImplementedError()

@app.command(
    name="metrics",
    help="Calculate metrics for existing SOM predictions.",
)
def metrics():
    raise NotImplementedError()
