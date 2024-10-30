import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import yaml
from lightning import Trainer
from lightning import seed_everything as lightning_seed_everything
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from awesom.create_dataset import LabeledData, UnlabeledData
from awesom.lightning_modules import GNN
from awesom.metrics_utils import TestLogger

warnings.filterwarnings("ignore", category=UserWarning)


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    lightning_seed_everything(seed)
    geometric_seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def load_data(input_path: str, mode: str) -> Dataset:
    """Loads labeled or unlabeled dataset based on mode."""
    if mode == "test":
        return LabeledData(root=input_path, transform=T.ToUndirected())
    elif mode == "infer":
        return UnlabeledData(root=input_path, transform=T.ToUndirected())
    else:
        raise ValueError("Mode must be either 'test' or 'infer'.")


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


def predict_with_ensemble(data, version_paths: List[Path]) -> tuple:
    """Run predictions for each model checkpoint in the ensemble."""
    num_samples = len(data)
    num_atoms = data.x.size(0)
    num_models = len(version_paths)
    device = torch.device("cpu")

    mol_id_ensemble = torch.empty(num_atoms, dtype=torch.int32, device=device)
    atom_id_ensemble = torch.empty(num_atoms, dtype=torch.int32, device=device)
    y_true_ensemble = torch.empty(num_atoms, dtype=torch.int32, device=device)
    logits_ensemble = torch.empty(
        (num_models, num_atoms), dtype=torch.float32, device=device
    )

    for i, version_path in enumerate(version_paths):
        checkpoint_files = list(Path(version_path, "checkpoints").glob("*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No .ckpt files found in {version_path}")

        model = load_model_from_checkpoint(checkpoint_files[0])

        trainer = Trainer(accelerator="auto", logger=False)
        prediction = trainer.predict(
            model=model,
            dataloaders=DataLoader(data, batch_size=num_samples, shuffle=False),
        )[0]

        logits, y_true, mol_id, atom_id = prediction

        if i == 0:
            mol_id_ensemble.copy_(mol_id)
            atom_id_ensemble.copy_(atom_id)
            y_true_ensemble.copy_(y_true)
        logits_ensemble[i] = logits

    return mol_id_ensemble, atom_id_ensemble, y_true_ensemble, logits_ensemble


def main():
    set_seeds()

    # Load data
    data = load_data(args.inputPath, args.mode)
    print(f"Loaded dataset with {len(data)} instances.")

    # Find model checkpoints
    version_paths = find_checkpoints(args.checkpointsPath)
    # Ensemble Prediction
    (
        mol_id_ensemble,
        atom_id_ensemble,
        y_true_ensemble,
        logits_ensemble,
    ) = predict_with_ensemble(data, version_paths)

    # Compute and log test results
    TestLogger.compute_and_log_test_results(
        mol_id_ensemble,
        atom_id_ensemble,
        y_true_ensemble,
        logits_ensemble,
        args.outputPath,
        args.mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Predicting SoMs for labeled (test) and unlabeled (infer) data."
    )

    parser.add_argument(
        "-i",
        dest="inputPath",
        type=str,
        required=True,
        help="Folder holding the input data.",
    )
    parser.add_argument(
        "-c",
        dest="checkpointsPath",
        type=str,
        required=True,
        help="Folder holding the model checkpoints.",
    )
    parser.add_argument(
        "-o",
        dest="outputPath",
        type=str,
        required=True,
        help="Folder to which the results will be written.",
    )
    parser.add_argument(
        "-m",
        dest="mode",
        type=str,
        required=True,
        choices=["test", "infer"],
        help="Mode: 'test' or 'infer'.",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    main()
    print("Finished in:", datetime.now() - start_time)
