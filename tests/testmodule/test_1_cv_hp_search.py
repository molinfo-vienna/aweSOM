import os
import warnings
from pathlib import Path
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
from awesom.models import M7
from awesom.training_module import GNN, train_model

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'DataFrame.swapaxes' is deprecated and will be removed in a future version.*",
)

INPUT_PATH: str = os.path.join(os.path.dirname(__file__), "test_data", "train")
OUTPUT_PATH: str = os.path.join(
    os.path.dirname(__file__), "test_output", "cv_hp_search"
)
HPARAMS_YAML_PATH: str = os.path.join(os.path.dirname(__file__))
MODEL: str = "M7"
EPOCHS: int = 10
NUM_CV_FOLDS: int = 2
NUM_OPTUNA_TRIALS: int = 2
BATCH_SIZE: int = 32


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    geometric_seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def save_hparams_to_yaml(path: Path, params: Dict[str, Any]) -> None:
    """Save hyperparameters to YAML file."""
    with open(path, "w") as f:
        yaml.dump(params, f, default_flow_style=False)


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
    hyperparams: Dict[str, Union[int, float]] = M7.get_params(trial)

    # K-fold cross validation
    kfold: KFold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_scores: List[float] = []

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
        model: GNN = GNN(data_params, hyperparams)

        # Train model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=int(hyperparams["epochs"]),
            patience=20,
        )

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

    return sum(fold_scores) / len(fold_scores)


def test_cv_hp_search() -> None:
    INPUT_PATH: str = os.path.join(os.path.dirname(__file__), "test_data", "train")
    OUTPUT_PATH: str = os.path.join(
        os.path.dirname(__file__), "test_output", "cv_hp_search"
    )
    EPOCHS: int = 10
    NUM_FOLDS: int = 2
    NUM_TRIALS: int = 2

    set_seeds()

    # Load data
    data: SOM = SOM(root=INPUT_PATH, transform=T.ToUndirected()).shuffle()
    print(f"Loaded {len(data)} instances")

    data_params: Dict[str, int] = {
        "num_node_features": data.num_node_features,
        "num_edge_features": data.num_edge_features,
    }

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Run Optuna optimization
    study: optuna.study.Study = optuna.create_study(
        direction="maximize", storage=f"sqlite:///{OUTPUT_PATH}/study.db"
    )

    study.optimize(
        lambda trial: objective(
            trial, data, data_params, NUM_FOLDS, EPOCHS, BATCH_SIZE
        ),
        n_trials=NUM_TRIALS,
    )

    # Save best hyperparameters
    best_params: Dict[str, Any] = study.best_trial.params
    best_params["epochs"] = EPOCHS

    with open(os.path.join(OUTPUT_PATH, "best_hparams.yaml"), "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {best_params}")

    print("Hyperparameter search completed successfully!")
