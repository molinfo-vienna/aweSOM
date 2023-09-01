import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

from collections import Counter
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statistics import mean
from typing import Dict, List, Tuple


__all__ = [
    "EarlyStopping",
    "seed_everything",
    "plot_losses",
    "save_predict",
]


"""
-------------------- Neural network related utility functions --------------------
"""


class EarlyStopping:
    """Early stops the training if validation loss doesn't
    improve after a given patience."""

    def __init__(
        self, patience: int = 5, delta: float = 0, verbose: bool = False
    ) -> None:
        """
        Args:
            patience (int): How many epcohs to wait after last time
                            validation loss improved before stopping.
                            Default: 5
            delta (float):  Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
        """
        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score = None
        self.early_stop: bool = False

    def __call__(self, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


"""
-------------------- General utility functions --------------------
"""


def seed_everything(seed: int = 42) -> None:
    """Seed os environment, python, numpy and torch for reproducibility.
    Args:
        seed (int): default 42
    Returns
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def plot_losses(
    train_loss: List[np.float64], val_loss: List[np.float64], dir: str
) -> None:
    """Plot training and validation losses.
    Args:
        train_loss (list): training losses
        val_loss(list): validation losses
        dir (str): output directory
    Returns:
        None
    """
    plt.figure()
    plt.plot(
        np.arange(0, len(train_loss), 1),
        train_loss,
        linestyle="-",
        linewidth=1,
        color="orange",
        label="Training Loss",
    )
    plt.plot(
        np.arange(0, len(train_loss), 1),
        val_loss,
        linestyle="-",
        linewidth=1,
        color="blue",
        label="Validation Loss",
    )
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dir)


def save_predict(
    outdir: str,
    y_trues: Dict[Tuple[int, int], float],
    y_preds: Dict[Tuple[int, int], List[float]],
    opt_thresholds: List[float],
) -> None:
    """Saves detailed prediction results to predictions.csv and performance summary to results.txt.
    Args:
        outdir (str): output directory
        y_trues (dict): true labels
        y_preds (dict): predicted SoM-probabilities ([0,1])
        opt_thresholds (List[float]): optimal discrimination thresholds
    Returns:
        None
    """

    # Compute binary labels (y_preds_bin) from SoM probabilities (y_preds)
    y_preds_bin = {
        key: [int(y_pred > threshold) for y_pred in preds]
        for key, preds in y_preds.items()
        for threshold in opt_thresholds
    }

    # Let the models in the ensemble classifier vote for the final binary label
    y_preds_voted = {
        key: Counter(preds).most_common(1)[0][0] for key, preds in y_preds_bin.items()
    }

    # Average SoM probability outputs
    y_preds_avg = [mean(preds) for preds in y_preds.values()]

    # Average optimal thresholds
    opt_thresholds_avg = np.round(np.average(opt_thresholds), 2)

    # Compute top1 and top2 accuracies
    mol_ids = np.unique([a for a, _ in y_trues.keys()])
    pred_top1 = []
    pred_top2 = []
    for mol_id in mol_ids:
        mask = [a == mol_id for a, b in y_trues.keys()]
        idx = np.argpartition([y_preds_avg[i] for i, x in enumerate(mask) if x], -1)[
            -1:
        ]
        if [list(y_trues.values())[i] for i, x in enumerate(mask) if x][idx[0]]:
            pred_top1.append(1)
        else:
            pred_top1.append(0)
        idx = np.argpartition([y_preds_avg[i] for i, x in enumerate(mask) if x], -2)[
            -2:
        ]
        if [list(y_trues.values())[i] for i, x in enumerate(mask) if x][idx[0]] or [
            list(y_trues.values())[i] for i, x in enumerate(mask) if x
        ][idx[1]]:
            pred_top2.append(1)
        else:
            pred_top2.append(0)
    top1 = np.sum(pred_top1) / len(mol_ids)
    top2 = np.sum(pred_top2) / len(mol_ids)

    results = {}
    y_true = list(y_trues.values())
    y_pred = list(y_preds_voted.values())
    results["mcc"] = round(matthews_corrcoef(y_true, y_pred), 2)
    results["precision"] = round(precision_score(y_true, y_pred), 2)
    results["recall"] = round(recall_score(y_true, y_pred), 2)
    if len(np.unique(np.array(y_true))) > 1:
        results["ROCAUC"] = round(roc_auc_score(y_true, y_preds_avg), 2)
    results["top1"] = round(top1, 2)
    results["top2"] = round(top2, 2)
    results["optThresholdAvg"] = round(opt_thresholds_avg, 2)

    with open(
        os.path.join(outdir, "results.txt"),
        "w",
        encoding="UTF8",
    ) as f:
        f.write(json.dumps(results))

    # Save molecular identifiers, atom identifiers, true labels,
    # predicted binary labels and (averaged) predicted som-probabilities of
    # single atoms to csv file
    rows = zip(
        [k for (k, _) in y_trues.keys()],
        [k for (_, k) in y_trues.keys()],
        list(y_trues.values()),
        list(y_preds_voted.values()),
        y_preds_avg,
    )
    with open(
        os.path.join(outdir, "predictions.csv"),
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "mol_id",
                "atom_id",
                "true_label",
                "predicted_binary_label",
                "averaged_som_probability",
            )
        )
        writer.writerows(rows)
