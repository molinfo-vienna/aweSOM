import argparse
import os
import warnings
from pathlib import Path
from typing import List

import torch
from torch_geometric import transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.metrics_utils import TestLogger
from awesom.training_module import predict_ensemble

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'DataFrame.swapaxes' is deprecated and will be removed in a future version.*",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Test ensemble model"
    )
    parser.add_argument("-i", "--input", required=True, help="Input data path")
    parser.add_argument(
        "-c", "--checkpoints", required=True, help="Model checkpoints path"
    )
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["test", "infer"],
        required=True,
        help="Test or inference mode",
    )
    args: argparse.Namespace = parser.parse_args()

    # Load data
    labeled: bool = args.mode == "test"
    data: SOM = SOM(root=args.input, labeled=labeled, transform=T.ToUndirected())

    print(f"Loaded {len(data)} instances for {args.mode}")

    # Find model checkpoints
    model_paths: List[str] = find_model_paths(args.checkpoints)
    print(f"Found {len(model_paths)} model checkpoints")

    # Create dataloader
    dataloader: DataLoader[Data] = DataLoader(data, batch_size=len(data), shuffle=False)

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
            logits_ensemble,
            y_trues,
            mol_ids,
            atom_ids,
            descriptions,
            args.output,
            args.mode,
        )

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
