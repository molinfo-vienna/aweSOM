import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Union

import torch
import yaml  # type: ignore
from lightning import Trainer
from lightning import seed_everything as lightning_seed_everything
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from awesom.create_dataset import LabeledData, UnlabeledData
from awesom.lightning_module import GNN
from awesom.metrics_utils import TestLogger

warnings.filterwarnings("ignore", category=UserWarning)

INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "test")
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), "output", "model")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "test")
MODE = "test"


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


def predict_with_ensemble(
    data: Dataset, version_paths: List[Path]
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
        predictions = trainer.predict(
            model=model,
            dataloaders=DataLoader(data, batch_size=num_molecules, shuffle=False),
        )

        if predictions is not None and len(predictions) > 0:
            predictions = predictions[0]
        else:
            raise ValueError("No predictions were made.")

        logits, y_true, mol_id, atom_id, description = predictions

        logits_ensemble[i] = torch.Tensor(logits)
        if i == 0:
            y_trues.copy_(torch.Tensor(y_true))
            mol_ids.copy_(torch.Tensor(mol_id))
            atom_ids.copy_(torch.Tensor(atom_id))

    return logits_ensemble, y_trues, mol_ids, atom_ids, description


def test_test() -> None:
    start_time = datetime.now()
    set_seeds()

    # Load data
    data = load_data(INPUT_PATH, MODE)
    print(f"Loaded dataset with {len(data)} instances.")

    # Record data and model loading time
    load_time = datetime.now() - start_time
    temptime = datetime.now()

    # Find model checkpoints
    version_paths = find_checkpoints(CHECKPOINTS_PATH)
    # Ensemble Prediction
    (
        logits_ensemble,
        y_trues,
        mol_ids,
        atom_ids,
        descriptions,
    ) = predict_with_ensemble(data, version_paths)

    # Record prediction time
    predict_time = datetime.now() - temptime
    temptime = datetime.now()

    # Compute and log test results
    TestLogger.compute_and_log_test_results(
        logits_ensemble,
        y_trues,
        mol_ids,
        atom_ids,
        descriptions,
        OUTPUT_PATH,
        MODE,
    )

    # Record logging time
    log_time = datetime.now() - temptime

    # Record total time
    total_time = datetime.now() - start_time
    print("Finished in:", datetime.now() - start_time)

    with open(OUTPUT_PATH + "/runtime.csv", "a") as f:
        f.write(f"{total_time},{load_time},{predict_time},{log_time}\n")
