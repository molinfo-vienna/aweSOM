import argparse
import random
import warnings
from datetime import datetime
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


if __name__ == "__main__":
    start_time = datetime.now()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser("Training the deep ensemble model.")

    parser.add_argument(
        "-i",
        dest="inputPath",
        type=str,
        required=True,
        help="Folder holding the input data.",
    )
    parser.add_argument(
        "-c",
        dest="hparamsYamlPath",
        type=str,
        required=True,
        help="Folder holding the yaml file with the optimal hyperparameters. \
            These should be determined prior to training by running the cv_hp_search.py script.",
    )
    parser.add_argument(
        "-o",
        dest="outputPath",
        type=str,
        required=True,
        help="Folder to which the output (trained model checkpoints, list of random seeds) will be written.",
    )

    args = parser.parse_args()

    # Load data
    data = SOM(root=args.inputPath, transform=T.ToUndirected())
    print(f"Loaded training data with {len(data)} instances.")
    data_params = dict(
        num_node_features=data.num_node_features,
        num_edge_features=data.num_edge_features,
        # num_mol_features=data.mol_x.shape[1],
    )

    random_seeds = random.sample(range(0, 1000), ENSEMBLE_SIZE)
    for seed in random_seeds:
        set_seeds(seed)
        hyperparams = load_hyperparams(args.hparamsYamlPath)
        logger = TensorBoardLogger(save_dir=args.outputPath, default_hp_metric=False)
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

    with open(Path(args.outputPath, "random_seeds.txt"), "w") as f:
        f.writelines(f"{seed}\n" for seed in random_seeds)

    print("Finished in:", datetime.now() - start_time)
