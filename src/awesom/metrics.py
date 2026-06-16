import csv
import os
from statistics import mean, stdev

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import RocCurveDisplay
from torchmetrics import AUROC, AveragePrecision, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryPrecision, BinaryRecall

THRESHOLD = 0.5


class MetricsCalculator:
    """Simple metrics calculator for site-of-metabolism prediction."""

    @staticmethod
    def compute_torchmetrics(
        y_probs: torch.Tensor, y_true: torch.Tensor
    ) -> dict[str, float]:
        """Compute all classification metrics."""
        y_pred = (y_probs >= THRESHOLD).int()

        device = y_probs.device

        return {
            "roc_auc": AUROC(task="binary").to(device)(y_probs, y_true).item(),
            "average_precision": AveragePrecision(task="binary")
            .to(device)(y_probs, y_true)
            .item(),
            "f1": F1Score(task="binary").to(device)(y_pred, y_true).item(),
            "matthew_corrcoef": MatthewsCorrCoef(task="binary").to(device)(y_pred, y_true).item(),
            "precision": BinaryPrecision().to(device)(y_pred, y_true).item(),
            "recall": BinaryRecall().to(device)(y_pred, y_true).item(),
        }

    @staticmethod
    def compute_ranking(y_probs: torch.Tensor, mol_ids: torch.Tensor) -> torch.Tensor:
        """Compute atom rankings within each molecule."""
        rankings = []
        for mol_id in torch.unique(mol_ids):
            mol_mask = mol_ids == mol_id
            mol_probs = y_probs[mol_mask]
            sorted_indices = torch.argsort(mol_probs, descending=True)
            ranks = torch.argsort(sorted_indices)
            rankings.append(ranks)
        return torch.cat(rankings)

    @staticmethod
    def compute_top2_accuracy(
        y_probs: torch.Tensor, y_true: torch.Tensor, mol_ids: torch.Tensor
    ) -> float:
        """Compute top-2 accuracy: fraction of molecules where at least one of top-2 predicted atoms is correct."""
        correct_molecules = 0
        total_molecules = 0

        for mol_id in torch.unique(mol_ids):
            mol_mask = mol_ids == mol_id
            mol_probs = y_probs[mol_mask]
            mol_true = y_true[mol_mask]

            top2_indices = torch.topk(mol_probs, min(2, len(mol_probs))).indices
            if torch.any(mol_true[top2_indices]):
                correct_molecules += 1
            total_molecules += 1

        return correct_molecules / total_molecules if total_molecules > 0 else 0.0
