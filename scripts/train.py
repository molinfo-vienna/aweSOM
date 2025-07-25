import argparse
import os
import random
import warnings
from typing import Dict, List, Union

import torch
import yaml  # type: ignore
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.gpu_utils import print_device_info
from awesom.model import SOMPredictor

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


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train ensemble model"
    )
    parser.add_argument("-i", "--input", required=True, help="Input data path")
    parser.add_argument(
        "-c", "--config", required=True, help="Hyperparameters YAML path"
    )
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--ensemble_size", type=int, default=10, help="Num. models in ensemble"
    )
    args: argparse.Namespace = parser.parse_args()

    print_device_info()

    print("Loading data...")
    data: SOM = SOM(root=args.input, transform=T.ToUndirected())
    print(f"Loaded {len(data)} training instances")

    data_params: Dict[str, int] = {
        "num_node_features": data.num_node_features,
        "num_edge_features": data.num_edge_features,
    }

    with open(os.path.join(args.config, "best_hparams.yaml"), "r") as f:
        hyperparams: Dict[str, Union[int, float]] = yaml.safe_load(f)

    print(f"\nTraining ensemble of {args.ensemble_size} models...")
    random_seeds: List[int] = random.sample(range(1000), args.ensemble_size)

    for i, seed in enumerate(random_seeds):
        print(f"Training model {i+1}/{args.ensemble_size} with seed {seed}")
        set_seed(seed)

        model: SOMPredictor = SOMPredictor(data_params, hyperparams)

        train_loader: DataLoader = DataLoader(
            data, batch_size=args.batch_size, shuffle=True
        )

        model_dir: str = os.path.join(args.output, f"model_{i}")
        log_dir: str = os.path.join(model_dir, "logs")
        checkpoint_dir: str = os.path.join(model_dir, "checkpoints")

        model.fit(
            train_loader=train_loader,
            max_epochs=int(hyperparams["epochs"]),
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            patience=20,
        )

    with open(os.path.join(args.output, "seeds.txt"), "w") as f:
        for seed in random_seeds:
            f.write(f"{seed}\n")

    print("Training completed!")


if __name__ == "__main__":
    main()
