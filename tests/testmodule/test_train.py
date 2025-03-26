import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore
from lightning import Trainer
from lightning import seed_everything as lightning_seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.lightning_module import GNN

warnings.filterwarnings("ignore", category=UserWarning)

BATCH_SIZE = 32
ENSEMBLE_SIZE = 10

INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "train")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "model")
HPARAMS_YAML_PATH = os.path.join(os.path.dirname(__file__))


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    lightning_seed_everything(seed)
    geometric_seed_everything(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_hyperparams(path: str) -> dict[str, Any]:
    """Load hyperparameters from YAML file."""
    with open(Path(path, "best_hparams.yaml"), "r") as file:
        return yaml.safe_load(file)


def test_train() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # Load data
    data = SOM(root=INPUT_PATH, transform=T.ToUndirected())
    print(f"Loaded training data with {len(data)} instances.")
    data_params = dict(
        num_node_features=data.num_node_features,
        num_edge_features=data.num_edge_features,
    )

    random_seeds = random.sample(range(0, 1000), ENSEMBLE_SIZE)
    for seed in random_seeds:
        set_seeds(seed)
        hyperparams = load_hyperparams(HPARAMS_YAML_PATH)
        logger = TensorBoardLogger(save_dir=OUTPUT_PATH, default_hp_metric=False)
        trainer = Trainer(
            accelerator="auto",
            max_epochs=hyperparams["epochs"],
            logger=logger,
            log_every_n_steps=1,
        )
        model = GNN(
            params=data_params,
            hyperparams=hyperparams,
            architecture=hyperparams["architecture"],
        )
        train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
        )

    with open(Path(OUTPUT_PATH, "random_seeds.txt"), "w") as f:
        f.writelines(f"{seed}\n" for seed in random_seeds)
