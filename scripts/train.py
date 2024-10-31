import argparse
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import yaml
from lightning import Trainer
from lightning import seed_everything as lightning_seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.lightning_module import GNN

warnings.filterwarnings("ignore", category=UserWarning)

BATCH_SIZE = 32
ENSEMBLE_SIZE = 50
VALIDATION_SET_SIZE = 0.1


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    lightning_seed_everything(seed)
    geometric_seed_everything(seed)


def prepare_data_loaders(data: Dataset) -> Tuple[DataLoader, DataLoader]:
    """Split data and prepare train/validation loaders."""
    train_data, val_data = train_test_split(
        data,
        test_size=VALIDATION_SET_SIZE,
        random_state=42,  # we always use the same seed here
    )
    print(
        f"Training instances: {len(train_data)}, Validation instances: {len(val_data)}"
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def load_hyperparams(path: str) -> dict:
    """Load hyperparameters from YAML file."""
    with open(Path(path, "best_hparams.yaml"), "r") as file:
        return yaml.safe_load(file)


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # Load data
    data = SOM(root=args.inputPath, transform=T.ToUndirected())
    print(f"Loaded dataset with {len(data)} instances.")
    data_params = dict(
        num_node_features=data.num_node_features,
        num_edge_features=data.num_edge_features,
        # num_mol_features=data.mol_x.shape[1],
    )

    random_seeds = random.sample(range(0, 1000), ENSEMBLE_SIZE)
    for seed in random_seeds:
        set_seeds(seed)
        train_loader, val_loader = prepare_data_loaders(data)
        hyperparams = load_hyperparams(args.hparamsYamlPath)
        hyperparams["mode"] = "ensemble"

        model = GNN(
            params=data_params,
            hyperparams=hyperparams,
            architecture=args.model,
        )

        logger = TensorBoardLogger(save_dir=args.outputPath, default_hp_metric=False)
        callbacks = [
            EarlyStopping(monitor="val/loss", mode="min", patience=20),
            ModelCheckpoint(monitor="val/loss", mode="min"),
        ]

        trainer = Trainer(
            accelerator="auto",
            max_epochs=args.epochs,
            logger=logger,
            log_every_n_steps=1,
            callbacks=callbacks,
        )

        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    with open(Path(args.outputPath, "random_seeds.txt"), "w") as f:
        f.writelines(f"{seed}\n" for seed in random_seeds)


if __name__ == "__main__":
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
        dest="epochs",
        type=int,
        required=True,
        default=1000,
        help="Maximum number of training epochs.",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    main()
    print("Finished in:", datetime.now() - start_time)
