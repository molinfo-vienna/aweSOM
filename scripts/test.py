import argparse
import os
import warnings
from pathlib import Path
from typing import List

from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.gpu_utils import print_device_info
from awesom.metrics_utils import log_results
from awesom.model import predict_ensemble

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

    print_device_info()

    # Load data
    labeled: bool = args.mode == "test"
    data: SOM = SOM(root=args.input, labeled=labeled, transform=T.ToUndirected())

    print(f"Loaded {len(data)} instances for {args.mode}")

    # Find model checkpoints
    model_paths: List[str] = find_model_paths(args.checkpoints)
    print(f"Found {len(model_paths)} model checkpoints")

    # Create dataloader
    dataloader: DataLoader = DataLoader(data, batch_size=len(data), shuffle=False)

    # Run ensemble predictions
    predictions = predict_ensemble(dataloader, model_paths)

    if predictions:
        # Log results using the unified function
        log_results(predictions.to("cpu"), args.output, args.mode)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
