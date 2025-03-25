import argparse
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from lightning import Trainer
from lightning import seed_everything as lightning_seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM, LabeledData
from awesom.lightning_module import GNN
from awesom.metrics_utils import TestLogger

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


def load_hyperparams(path: str) -> dict:
    """Load hyperparameters from YAML file."""
    with open(Path(path, "best_hparams.yaml"), "r") as file:
        return yaml.safe_load(file)


def find_checkpoints(checkpoints_path: str) -> List[Path]:
    """Retrieve sorted list of checkpoint paths."""
    checkpoints_dir = Path(checkpoints_path, "lightning_logs")
    version_paths = sorted(
        [p for p in checkpoints_dir.glob("version_*")],
        key=lambda x: int(x.stem.split("_")[1]),
    )
    if not version_paths:
        raise FileNotFoundError(f"No checkpoint versions found in {checkpoints_path}")
    return version_paths


def load_model_from_checkpoint(checkpoint_path: Path) -> GNN:
    """Load model and its hyperparameters from checkpoint."""
    hyperparams = yaml.safe_load(
        Path(checkpoint_path.parent.parent, "hparams.yaml").read_text()
    )
    model = GNN.load_from_checkpoint(checkpoint_path, **hyperparams)
    return model


def predict_with_ensemble(
    data, version_paths: List[Path]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Run predictions for each model checkpoint in the ensemble."""
    num_molecules = len(data)
    num_atoms = data.x.size(0)
    num_models = len(version_paths)
    device = torch.device("cpu")
    # switching to CPU is necessary because "descriptions"
    # (a.k.a. the molecular identifiers of the input SD-file)
    # is a list of strings, which is not supported on GPU

    logits_ensemble = torch.empty(
        (num_models, num_atoms), dtype=torch.float32, device=device
    )
    y_trues = torch.empty(num_atoms, dtype=torch.int32, device=device)
    mol_ids = torch.empty(num_molecules, dtype=torch.int32, device=device)
    atom_ids = torch.empty(num_atoms, dtype=torch.int32, device=device)

    for i, version_path in enumerate(version_paths):
        checkpoint_files = list(Path(version_path, "checkpoints").glob("*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No .ckpt files found in {version_path}")

        model = load_model_from_checkpoint(checkpoint_files[0])

        trainer = Trainer(accelerator="auto", logger=False)
        prediction = trainer.predict(
            model=model,
            dataloaders=DataLoader(data, batch_size=num_molecules, shuffle=False),
        )[0]

        logits, y_true, mol_id, atom_id, description = prediction

        logits_ensemble[i] = logits
        if i == 0:
            y_trues.copy_(y_true)
            mol_ids.copy_(mol_id)
            atom_ids.copy_(atom_id)

    return logits_ensemble, y_trues, mol_ids, atom_ids, description


def main() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    test_data = LabeledData(root=args.testingDataPath, transform=T.ToUndirected())
    print(f"Loaded test data with {len(test_data)} instances.")

    train_data = SOM(root=args.trainingDataPath, transform=T.ToUndirected())
    print(f"Loaded training data with {len(train_data)} instances.")

    data_params = dict(
        num_node_features=train_data.num_node_features,
        num_edge_features=train_data.num_edge_features,
    )

    # Train ensemble models with varying proportions of training data
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

        output_path_p = os.path.join(args.outputPath, str(int(p * 100)))
        output_path_model = os.path.join(output_path_p, "model")
        output_path_test = os.path.join(output_path_p, "test")

        # Select subset of training data
        if p < 1:
            sub_train_data, _ = train_test_split(train_data, test_size=(1 - p))
        else:
            sub_train_data = train_data

        # Train ensemble models with different random seeds
        random_seeds = random.sample(range(0, 1000), ENSEMBLE_SIZE)
        for seed in random_seeds:
            set_seeds(seed)
            hyperparams = load_hyperparams(args.hparamsYamlPath)
            logger = TensorBoardLogger(
                save_dir=output_path_model, default_hp_metric=False
            )
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
            train_loader = DataLoader(
                sub_train_data, batch_size=BATCH_SIZE, shuffle=True
            )
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
            )

        with open(Path(output_path_model, "random_seeds.txt"), "w") as f:
            f.writelines(f"{seed}\n" for seed in random_seeds)

        # Evaluate ensemble models on the test set
        set_seeds(42)
        # Find model checkpoints
        version_paths = find_checkpoints(output_path_model)
        # Ensemble Prediction
        (
            logits_ensemble,
            y_trues,
            mol_ids,
            atom_ids,
            descriptions,
        ) = predict_with_ensemble(test_data, version_paths)

        # Compute and log test results
        TestLogger.compute_and_log_test_results(
            logits_ensemble,
            y_trues,
            mol_ids,
            atom_ids,
            descriptions,
            output_path_test,
            "test",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training the deep ensemble model.")

    parser.add_argument(
        "-train",
        dest="trainingDataPath",
        type=str,
        required=True,
        help="Folder holding the training data.",
    )
    parser.add_argument(
        "-test",
        dest="testingDataPath",
        type=str,
        required=True,
        help="Folder holding the testing data.",
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
        help="Folder to which the output will be written.",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    main()
    print("Finished in:", datetime.now() - start_time)
