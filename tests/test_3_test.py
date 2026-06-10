import os
from pathlib import Path
from typing import List

import torch
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.metrics_utils import ResultsLogger
from awesom.model import predict_ensemble


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

    labeled: bool = MODE == "test"
    data: SOM = SOM(root=INPUT_PATH, labeled=labeled, transform=T.ToUndirected())

    print(f"Loaded {len(data)} instances for {MODE}")

    model_paths: List[str] = find_model_paths(CHECKPOINTS_PATH)
    print(f"Found {len(model_paths)} model checkpoints")

    dataloader: DataLoader = DataLoader(data, batch_size=len(data), shuffle=False)

    predictions = predict_ensemble(dataloader, model_paths)

    if predictions:
        results_logger = ResultsLogger(OUTPUT_PATH)
        results_logger.save_results(predictions.to(torch.device("cpu")), MODE)

    print(f"Results saved to {OUTPUT_PATH}")
