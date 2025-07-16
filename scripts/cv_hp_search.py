import argparse
import csv
import os
import warnings
from statistics import mean, stdev
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
from awesom.metrics_utils import MetricsCalculator
from awesom.model import GINEWithContextPooling, SOMPredictor

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'DataFrame.swapaxes' is deprecated and will be removed in a future version.*",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

THRESHOLD = 0.5


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    geometric_seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")


def save_fold_predictions(
    fold_predictions: Dict[str, List],
    output_dir: str,
    fold: int,
) -> None:
    """Save detailed predictions for a single fold to CSV."""
    csv_path = os.path.join(output_dir, f"validation_fold_{fold}.csv")
    
    headers = [
        "mol_id",
        "atom_id", 
        "y_true",
        "y_prob",
        "y_pred",
        "ranking",
    ]
    
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in zip(*fold_predictions.values()):
            writer.writerow(row)


def save_validation_metrics(
    all_fold_metrics: Dict[str, List[float]],
    output_dir: str,
) -> None:
    """Save validation metrics with standard deviations to text file."""
    txt_path = os.path.join(output_dir, "validation.txt")
    
    with open(txt_path, "w") as f:
        for metric_name, values in all_fold_metrics.items():
            mean_val = mean(values)
            std_val = stdev(values) if len(values) > 1 else 0.0
            f.write(f"{metric_name}: {round(mean_val, 4)} +/- {round(std_val, 4)}\n")


def objective(
    trial: optuna.trial.Trial,
    data: Dataset,
    data_params: Dict[str, int],
    num_folds: int,
    max_epochs: int,
    batch_size: int,
    output_dir: str,
) -> float:
    """Optuna objective function for hyperparameter optimization."""

    # Get hyperparameters from trial
    hyperparams: Dict[str, Union[int, float]] = GINEWithContextPooling.get_params(trial)
    hyperparams["epochs"] = max_epochs

    # K-fold cross validation
    kfold: KFold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_scores: List[float] = []
    fold_epochs: List[int] = []  # Track epochs for each fold
    
    # Store metrics across all folds for final statistics
    all_fold_metrics: Dict[str, List[float]] = {
        "ROC-AUC": [],
        "PR-AUC": [],
        "F1": [],
        "MCC": [],
        "Precision": [],
        "Recall": [],
        "Top-2": [],
    }
    
    metrics_calc = MetricsCalculator()

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

        # Use the modified fit method that returns the actual number of epochs
        actual_epochs = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            patience=20,
        )

        fold_epochs.append(actual_epochs)
        print(f"  Fold {fold + 1} stopped at epoch {actual_epochs}")

        # Evaluate on validation set and collect predictions
        model.eval()
        val_losses: List[float] = []
        val_mccs: List[float] = []
        
        # Collect all predictions for this fold
        all_logits: List[torch.Tensor] = []
        all_y_trues: List[torch.Tensor] = []
        all_mol_ids: List[torch.Tensor] = []
        all_atom_ids: List[torch.Tensor] = []
        all_descriptions: List[str] = []

        with torch.no_grad():
            for batch in val_loader:
                loss, mcc = model.val_step(batch)
                val_losses.append(loss)
                val_mccs.append(mcc)
                
                # Get detailed predictions
                logits, y_true, mol_id, atom_id, descriptions = model.predict(batch)
                all_logits.append(logits)
                all_y_trues.append(y_true)
                all_mol_ids.append(mol_id)
                all_atom_ids.append(atom_id)
                all_descriptions.extend(descriptions)

        # Concatenate all predictions for this fold
        fold_logits = torch.cat(all_logits, dim=0)
        fold_y_trues = torch.cat(all_y_trues, dim=0)
        fold_mol_ids = torch.cat(all_mol_ids, dim=0)
        fold_atom_ids = torch.cat(all_atom_ids, dim=0)
        fold_y_probs = torch.sigmoid(fold_logits)
        
        # Compute rankings
        rankings = metrics_calc.compute_ranking(fold_y_probs, fold_mol_ids)
        
        # Compute all metrics for this fold
        fold_metrics = metrics_calc.compute_torchmetrics(fold_y_probs, fold_y_trues)
        fold_top2 = metrics_calc.compute_top2_accuracy(fold_y_probs, fold_y_trues, fold_mol_ids)
        
        # Store metrics
        for metric_name, value in fold_metrics.items():
            all_fold_metrics[metric_name].append(value)
        all_fold_metrics["Top-2"].append(fold_top2)

        avg_mcc: float = sum(val_mccs) / len(val_mccs)
        fold_scores.append(avg_mcc)
        print(f"  Fold {fold + 1} MCC: {avg_mcc:.3f}")
        
        # Save detailed predictions for this fold
        fold_predictions = {
            "mol_id": all_descriptions,
            "atom_id": fold_atom_ids.tolist(),
            "y_true": fold_y_trues.tolist(),
            "y_prob": [round(p, 4) for p in fold_y_probs.tolist()],
            "y_pred": [round(p, 4) for p in (fold_y_probs >= THRESHOLD).int().tolist()],
            "ranking": rankings.tolist(),
        }
        save_fold_predictions(fold_predictions, output_dir, fold)

    # Store the average optimal epochs in the trial user attributes
    avg_optimal_epochs = int(sum(fold_epochs) / len(fold_epochs))
    trial.set_user_attr("optimal_epochs", avg_optimal_epochs)
    
    # Save validation metrics with standard deviations
    save_validation_metrics(all_fold_metrics, output_dir)

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
            trial, data, data_params, args.folds, args.epochs, args.batch_size, args.output
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
