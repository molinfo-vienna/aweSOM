import argparse
import os
import warnings
from typing import Any, Dict, List, Union

import optuna
import torch
import yaml  # type: ignore
from sklearn.model_selection import KFold
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from awesom.create_dataset import SOM
from awesom.gpu_utils import print_device_info
from awesom.model import GINEWithContextPooling, SOMPredictor

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'DataFrame.swapaxes' is deprecated and will be removed in a future version.*",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    geometric_seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def objective(
    trial: optuna.trial.Trial,
    data: Dataset,
    data_params: Dict[str, int],
    num_folds: int,
    max_epochs: int,
    batch_size: int,
) -> float:
    """Optuna objective function for hyperparameter optimization."""

    # Get hyperparameters from trial
    hyperparams: Dict[str, Union[int, float]] = GINEWithContextPooling.get_params(trial)
    hyperparams["epochs"] = max_epochs

    # K-fold cross validation
    kfold: KFold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_scores: List[float] = []
    fold_epochs: List[int] = []  # Track epochs for each fold

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"Trial {trial.number}, Fold {fold + 1}/{num_folds}")

        # Split data
        train_data: Dataset = data[train_idx]
        val_data: Dataset = data[val_idx]

        # Create dataloaders
        train_loader: DataLoader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        val_loader: DataLoader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )

        # Create model
        model: SOMPredictor = SOMPredictor(data_params, hyperparams)

        # Use the modfied fit method that returs the actual number of epochs
        actual_epochs = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            patience=20,
        )

        fold_epochs.append(actual_epochs)
        print(f"  Fold {fold + 1} stopped at epoch {actual_epochs}")

        # Evaluate on validation set
        model.eval()
        val_losses: List[float] = []
        val_mccs: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                loss, mcc = model.val_step(batch)
                val_losses.append(loss)
                val_mccs.append(mcc)

        avg_mcc: float = sum(val_mccs) / len(val_mccs)
        fold_scores.append(avg_mcc)
        print(f"  Fold {fold + 1} MCC: {avg_mcc:.3f}")

    # Store the average optimal epochs in the trial user attributes
    avg_optimal_epochs = int(sum(fold_epochs) / len(fold_epochs))
    trial.set_user_attr("optimal_epochs", avg_optimal_epochs)

    return sum(fold_scores) / len(fold_scores)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Hyperparameter search with cross-validation"
    )
    parser.add_argument("-i", "--input", required=True, help="Input data path")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum epochs")
    parser.add_argument("--folds", type=int, default=10, help="Num. CV folds")
    parser.add_argument("--trials", type=int, default=20, help="Num. Optuna trials")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args: argparse.Namespace = parser.parse_args()

    set_seeds()
    print_device_info()

    # Load data
    print("Loading data...")
    data: SOM = SOM(root=args.input, transform=T.ToUndirected()).shuffle()
    print(f"Loaded {len(data)} instances")

    data_params: Dict[str, int] = {
        "num_node_features": data.num_node_features,
        "num_edge_features": data.num_edge_features,
    }

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run Optuna optimization
    print(
        f"\nStarting hyperparameter search with {args.trials} trials and {args.folds}-fold CV..."
    )

    study: optuna.study.Study = optuna.create_study(
        direction="maximize",
        load_if_exists=True,
        storage=f"sqlite:///{args.output}/study.db",
        study_name="cv_hp_search",
    )

    study.optimize(
        lambda trial: objective(
            trial, data, data_params, args.folds, args.epochs, args.batch_size
        ),
        n_trials=args.trials,
    )

    # Save best hyperparameters
    best_params: Dict[str, Any] = study.best_trial.params
    # Additionally store the optimal epochs from the best trial
    best_params["epochs"] = study.best_trial.user_attrs["optimal_epochs"]

    with open(os.path.join(args.output, "best_hparams.yaml"), "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    print(f"\nBest trial: {study.best_trial.value:.3f}")
    print(f"Best hyperparameters: {best_params}")


if __name__ == "__main__":
    main()
