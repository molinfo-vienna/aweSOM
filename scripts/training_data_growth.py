import argparse
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Union

import torch
import yaml  # type: ignore
from sklearn.model_selection import train_test_split
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from awesom.create_dataset import SOM
from awesom.gpu_utils import print_device_info
from awesom.metrics_utils import ResultsLogger
from awesom.model import SOMPredictor, predict_ensemble

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'DataFrame.swapaxes' is deprecated and will be removed in a future version.*",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)


def find_model_paths(checkpoints_path: str) -> List[str]:
    """Find all model checkpoint paths."""
    checkpoints_dir: Path = Path(checkpoints_path)
    model_paths: List[str] = []

    for model_dir in sorted(checkpoints_dir.glob("model_*")):
        checkpoint_path: Path = model_dir / "checkpoints" / "best_model.ckpt"
        if checkpoint_path.exists():
            model_paths.append(str(checkpoint_path))

    return model_paths


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Training data growth experiment"
    )
    parser.add_argument("--train", required=True, help="Training data path")
    parser.add_argument("--test", required=True, help="Test data path")
    parser.add_argument("--config", required=True, help="Hyperparameters YAML path")
    parser.add_argument("--output", required=True, help="Output path")
    args: argparse.Namespace = parser.parse_args()

    print_device_info()

    # Load data
    print("Loading data...")
    train_data: SOM = SOM(root=args.train, labeled=True, transform=T.ToUndirected())
    test_data: SOM = SOM(root=args.test, labeled=True, transform=T.ToUndirected())
    print(f"Loaded {len(train_data)} training and {len(test_data)} test instances")

    # Load hyperparameters
    with open(args.config, "r") as f:
        hyperparams: Dict[str, Union[int, float]] = yaml.safe_load(f)

    data_params: Dict[str, int] = {
        "num_node_features": train_data.num_node_features,
        "num_edge_features": train_data.num_edge_features,
    }

    # Test different proportions of training data
    proportions: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(
        f"\nStarting training data growth experiment with {len(proportions)} proportions..."
    )

    for p in tqdm(proportions, desc="Data Proportions"):
        # Select subset of training data
        subset_data: Dataset
        if p < 1.0:
            subset_size: int = int(len(train_data) * p)
            subset_data, _ = train_test_split(train_data, train_size=subset_size)
        else:
            subset_data = train_data

        # Train ensemble
        random_seeds: List[int] = random.sample(range(1000), 10)

        for i, seed in enumerate(
            tqdm(random_seeds, desc=f"Training {p*100}%", leave=False)
        ):
            set_seed(seed)

            # Create model and train
            model: SOMPredictor = SOMPredictor(data_params, hyperparams)
            train_loader: DataLoader = DataLoader(
                subset_data, batch_size=32, shuffle=True
            )

            model_dir: str = os.path.join(args.output, f"{int(p*100)}", f"model_{i}")
            log_dir: str = os.path.join(model_dir, "logs")
            checkpoint_dir: str = os.path.join(model_dir, "checkpoints")

            # Train using the new fit method
            model.fit(
                train_loader=train_loader,
                max_epochs=int(hyperparams["epochs"]),
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir,
                patience=20,
            )

        # Evaluate ensemble on test set
        set_seed(42)

        model_paths: List[str] = find_model_paths(
            os.path.join(args.output, f"{int(p*100)}")
        )
        test_loader: DataLoader = DataLoader(
            test_data, batch_size=len(test_data), shuffle=False
        )

        ensemble_predictions = predict_ensemble(test_loader, model_paths)

        if ensemble_predictions:
            # Save results using the unified function
            output_dir: str = os.path.join(args.output, f"{int(p*100)}", "test")
            results_logger = ResultsLogger(output_dir)
            results_logger.save_results(ensemble_predictions, output_dir, "test")

    print("\nTraining data growth experiment completed!")


if __name__ == "__main__":
    main()
