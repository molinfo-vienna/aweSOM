import os
import warnings
from pathlib import Path
from typing import List

import torch
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.metrics_utils import TestLogger
from awesom.training_module import predict_ensemble

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'DataFrame.swapaxes' is deprecated and will be removed in a future version.*",
)


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    geometric_seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def find_model_paths(checkpoints_path: str) -> List[str]:
    """Find all model checkpoint paths."""
    checkpoints_dir: Path = Path(checkpoints_path)
    model_paths: List[str] = []

    for model_dir in sorted(checkpoints_dir.glob("model_*")):
        checkpoint_path: Path = model_dir / "checkpoints" / "best_model.ckpt"
        if checkpoint_path.exists():
            model_paths.append(str(checkpoint_path))

    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoints_path}")

    return model_paths


def test_test() -> None:
    INPUT_PATH: str = os.path.join(os.path.dirname(__file__), "test_data", "test")
    CHECKPOINTS_PATH: str = os.path.join(
        os.path.dirname(__file__), "test_output", "model"
    )
    OUTPUT_PATH: str = os.path.join(os.path.dirname(__file__), "test_output", "test")
    MODE: str = "test"

    # Load data
    labeled: bool = MODE == "test"
    data: SOM = SOM(root=INPUT_PATH, labeled=labeled, transform=T.ToUndirected())

    print(f"Loaded {len(data)} instances for {MODE}")

    # Find model checkpoints
    model_paths: List[str] = find_model_paths(CHECKPOINTS_PATH)
    print(f"Found {len(model_paths)} model checkpoints")

    # Create dataloader
    dataloader: DataLoader = DataLoader(data, batch_size=len(data), shuffle=False)

    # Run ensemble predictions
    ensemble_predictions = predict_ensemble(dataloader, model_paths)

    # Process predictions (assuming single batch)
    if ensemble_predictions and ensemble_predictions[0]:
        # Extract predictions from first batch
        logits_ensemble: torch.Tensor = torch.stack(
            [pred[0][0] for pred in ensemble_predictions]
        )
        y_trues: torch.Tensor = ensemble_predictions[0][0][1]
        mol_ids: torch.Tensor = ensemble_predictions[0][0][2]
        atom_ids: torch.Tensor = ensemble_predictions[0][0][3]
        descriptions: List[str] = ensemble_predictions[0][0][4]

        # Compute and log results
        TestLogger.compute_and_log_test_results(
            logits_ensemble, y_trues, mol_ids, atom_ids, descriptions, OUTPUT_PATH, MODE
        )

    print(f"Results saved to {OUTPUT_PATH}")
