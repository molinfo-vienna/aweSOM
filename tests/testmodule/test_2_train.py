import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import yaml  # type: ignore
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.model import SOMPredictor


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    geometric_seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def test_train() -> None:
    BATCH_SIZE: int = 32
    ENSEMBLE_SIZE: int = 10
    INPUT_PATH: str = os.path.join(os.path.dirname(__file__), "test_data", "train")
    OUTPUT_PATH: str = os.path.join(os.path.dirname(__file__), "test_output", "model")
    CONFIG_PATH: str = os.path.join(
        os.path.dirname(__file__), "test_output", "cv_hp_search", "best_hparams.yaml"
    )

    # Load data
    data: SOM = SOM(root=INPUT_PATH, transform=T.ToUndirected())
    print(f"Loaded {len(data)} training instances")

    data_params: Dict[str, int] = {
        "num_node_features": data.num_node_features,
        "num_edge_features": data.num_edge_features,
    }

    # Load hyperparameters
    with open(CONFIG_PATH, "r") as f:
        hyperparams: Dict[str, Union[int, float]] = yaml.safe_load(f)

    # Train ensemble
    random_seeds: List[int] = random.sample(range(1000), ENSEMBLE_SIZE)

    for i, seed in enumerate(random_seeds):
        print(f"Training model {i+1}/{ENSEMBLE_SIZE}")
        set_seeds(seed)

        # Create model
        model: SOMPredictor = SOMPredictor(data_params, hyperparams)

        # Create dataloader
        train_loader: DataLoader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

        # Setup output directories
        model_dir: str = os.path.join(OUTPUT_PATH, f"model_{i}")
        log_dir: str = os.path.join(model_dir, "logs")
        checkpoint_dir: str = os.path.join(model_dir, "checkpoints")

        # Train using the new fit method
        model.fit(
            train_loader=train_loader,
            max_epochs=int(hyperparams["epochs"]),
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            patience=20,
        )

    # Save seeds
    with open(os.path.join(OUTPUT_PATH, "seeds.txt"), "w") as f:
        for seed in random_seeds:
            f.write(f"{seed}\n")

    print("Training completed!")
