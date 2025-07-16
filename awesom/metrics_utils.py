import csv
import os
from statistics import mean, stdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import RocCurveDisplay
from torchmetrics import AUROC, AveragePrecision, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryPrecision, BinaryRecall

from .model import EnsemblePredictions

THRESHOLD = 0.5
NUM_BOOTSTRAPS = 1000


class MetricsCalculator:
    """Simple metrics calculator for site-of-metabolism prediction."""

    @staticmethod
    def compute_torchmetrics(
        y_probs: torch.Tensor, y_true: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all classification metrics."""
        y_pred = (y_probs >= THRESHOLD).int()

        return {
            "ROC-AUC": AUROC(task="binary")(y_probs, y_true).item(),
            "PR-AUC": AveragePrecision(task="binary")(y_probs, y_true).item(),
            "F1": F1Score(task="binary")(y_pred, y_true).item(),
            "MCC": MatthewsCorrCoef(task="binary")(y_pred, y_true).item(),
            "Precision": BinaryPrecision()(y_pred, y_true).item(),
            "Recall": BinaryRecall()(y_pred, y_true).item(),
        }

    @staticmethod
    def compute_ranking(y_probs: torch.Tensor, mol_ids: torch.Tensor) -> torch.Tensor:
        """Compute atom rankings within each molecule."""
        rankings = []
        for mol_id in torch.unique(mol_ids):
            mol_mask = mol_ids == mol_id
            mol_probs = y_probs[mol_mask]
            # Sort by probability (descending) and get ranks
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

            # Get top 2 predictions
            top2_indices = torch.topk(mol_probs, min(2, len(mol_probs))).indices
            if torch.any(mol_true[top2_indices]):
                correct_molecules += 1
            total_molecules += 1

        return correct_molecules / total_molecules if total_molecules > 0 else 0.0


class ResultsLogger:
    """Simple logger for saving prediction results and metrics."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        self.metrics_calc = MetricsCalculator()

    def save_metrics(
        self,
        bootstrap_metrics: Dict[str, List[float]],
    ) -> None:
        """Save metrics and bootstrap confidence intervals to text file."""
        with open(os.path.join(self.output_path, "results.txt"), "w") as f:
            for metric_name, values in bootstrap_metrics.items():
                mean_val = mean(values)
                std_val = stdev(values) if len(values) > 1 else 0.0
                f.write(
                    f"{metric_name}: {round(mean_val, 4)} +/- {round(std_val, 4)}\n"
                )

    def save_predictions(
        self,
        atom_ids: torch.Tensor,  # (num_atoms,)
        mol_ids: torch.Tensor,  # (num_atoms,)
        y_trues: torch.Tensor,  # (num_atoms,)
        y_probs: torch.Tensor,  # (num_atoms,)
        rankings: torch.Tensor,  # (num_atoms,)
        uncertainties: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (num_atoms,)
        descriptions: List[str],  # (num_molecules,)
    ) -> None:
        """Save detailed predictions to CSV file."""
        y_preds = (y_probs >= THRESHOLD).int()
        u_ale, u_epi, u_tot = uncertainties

        # Blow up the descriptions to match the number of atoms
        # Each molecule has a different number of atoms, which
        # can be extracted by counting the number of repeating
        # identical entries in the mol_ids tensor.
        counts = torch.bincount(mol_ids)
        expanded_descriptions = [
            desc
            for desc, count in zip(descriptions, counts.tolist())
            for _ in range(count)
        ]

        # Prepare CSV headers and data
        headers = [
            "mol_id",
            "atom_id",
            "y_true",
            "y_prob",
            "y_pred",
            "ranking",
            "u_ale",
            "u_epi",
            "u_tot",
        ]
        data = [
            expanded_descriptions,
            atom_ids.tolist(),
            y_trues.tolist(),
            [round(p, 4) for p in y_probs.tolist()],
            [round(p, 4) for p in y_preds.tolist()],
            rankings.tolist(),
            [round(u, 4) for u in u_ale.tolist()],
            [round(u, 4) for u in u_epi.tolist()],
            [round(u, 4) for u in u_tot.tolist()],
        ]

        # Write CSV
        with open(os.path.join(self.output_path, "results.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in zip(*data):
                writer.writerow(row)


    def save_roc_curve(self, y_true: torch.Tensor, y_probs: torch.Tensor) -> None:
        """Save ROC curve plot."""
        RocCurveDisplay.from_predictions(y_true, y_probs)
        plt.savefig(
            os.path.join(self.output_path, "roc.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()


    def compute_bootstrap_metrics(
        self,
        y_probs: torch.Tensor,
        y_trues: torch.Tensor,
        mol_ids: torch.Tensor,
        n_bootstrap: int = NUM_BOOTSTRAPS,
    ) -> Dict[str, List[float]]:
        """Compute metrics with bootstrap confidence intervals."""
        bootstrap_results: Dict[str, List[float]] = {
            metric: []
            for metric in ["ROC-AUC", "PR-AUC", "F1", "MCC", "Precision", "Recall", "Top-2"]
        }

        for _ in range(n_bootstrap):
            sampled_mol_ids = mol_ids[torch.randint(len(mol_ids), (len(mol_ids),))]
            mask = torch.isin(mol_ids, sampled_mol_ids)

            y_probs_sample = y_probs[mask]
            y_trues_sample = y_trues[mask]
            mol_ids_sample = mol_ids[mask]

            metrics = self.metrics_calc.compute_torchmetrics(y_probs_sample, y_trues_sample)
            top2 = self.metrics_calc.compute_top2_accuracy(
                y_probs_sample, y_trues_sample, mol_ids_sample
            )

            for metric_name, value in metrics.items():
                bootstrap_results[metric_name].append(value)
            bootstrap_results["Top-2"].append(top2)

        return bootstrap_results


    def save_results(
        self,
        predictions: EnsemblePredictions,
        mode: str = "inference",
    ) -> None:
        """
        Logs results for both test and inference scenarios.

        Args:
            predictions: EnsemblePredictions object
            output_path: where to save results
            mode: "test" or "inference"
        """
        # Compute average probabilities from logits
        y_probs = torch.mean(predictions.get_probabilities(), dim=0)

        # Compute uncertainties
        uncertainties = predictions.get_uncertainties()

        # Compute rankings
        rankings = self.metrics_calc.compute_ranking(y_probs, predictions.mol_ids)

        # Save predictions
        self.save_predictions(
            atom_ids=predictions.atom_ids,
            mol_ids=predictions.mol_ids,
            y_trues=(
                predictions.y_trues
                if mode == "test"
                else torch.zeros_like(predictions.y_trues)
            ),
            y_probs=y_probs,
            rankings=rankings,
            uncertainties=uncertainties,
            descriptions=predictions.descriptions,
        )

        if mode == "test":
            bootstrap_metrics = self.compute_bootstrap_metrics(
                y_probs, predictions.y_trues, predictions.mol_ids
            )
            self.save_metrics(bootstrap_metrics)
            self.save_roc_curve(predictions.y_trues, y_probs)
