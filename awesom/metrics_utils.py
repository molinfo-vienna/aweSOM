import csv
import os
from statistics import mean, stdev
from typing import List

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import RocCurveDisplay
from torchmetrics import AUROC, AveragePrecision, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryPrecision, BinaryRecall

NUM_BOOTSTRAPS = 1000
THRESHOLD = 0.5


class BaseMetrics:
    @classmethod
    def compute_ranking(
        cls, y_probs: torch.Tensor, mol_ids: torch.Tensor
    ) -> torch.Tensor:
        ranking = torch.cat(
            [
                torch.argsort(
                    torch.argsort(
                        torch.index_select(
                            y_probs, 0, torch.where(mol_ids == mol_id)[0]
                        ),
                        dim=0,
                        descending=True,
                    ),
                    dim=0,
                    descending=False,
                )
                for mol_id in list(dict.fromkeys(mol_ids.tolist()))
            ]
        )
        return ranking

    @classmethod
    def compute_shannon_entropy(cls, p: torch.Tensor) -> torch.Tensor:
        return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))

    @classmethod
    def compute_uncertainties(
        cls, y_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_tot = cls.compute_shannon_entropy(torch.mean(y_probs, dim=0))
        u_ale = torch.mean(cls.compute_shannon_entropy(y_probs), dim=0)

        u_epi = u_tot - u_ale

        return u_ale, u_epi, u_tot

    @classmethod
    def compute_auroc(cls, y_prob: torch.Tensor, y_true: torch.Tensor) -> float:
        auroc = AUROC(task="binary")
        return auroc(y_prob, y_true).item()

    @classmethod
    def compute_average_precision(
        cls, y_prob: torch.Tensor, y_true: torch.Tensor
    ) -> float:
        average_precision = AveragePrecision(task="binary")
        return average_precision(y_prob, y_true).item()

    @classmethod
    def compute_f1(cls, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        f1 = F1Score(task="binary")
        return f1(y_pred, y_true).item()

    @classmethod
    def compute_mcc(cls, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        mcc = MatthewsCorrCoef(task="binary")
        return mcc(y_pred, y_true).item()

    @classmethod
    def compute_precision(cls, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        precision = BinaryPrecision()
        return precision(y_pred, y_true).item()

    @classmethod
    def compute_recall(cls, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        recall = BinaryRecall()
        return recall(y_pred, y_true).item()

    @classmethod
    def compute_top2(
        cls,
        y_probs: torch.Tensor,
        y_trues: torch.Tensor,
        mol_ids_expanded: torch.Tensor,
    ) -> float:
        top2_counter: int = 0
        for id in list(
            dict.fromkeys(mol_ids_expanded.tolist())
        ):  # This is a somewhat complicated way to get an ordered set (masked_sorted_y), but it works.
            mask = torch.where(mol_ids_expanded == id)[0]
            masked_y_trues = y_trues[mask]
            masked_y_probs = y_probs[mask]
            masked_sorted_y_trues = torch.index_select(
                masked_y_trues,
                dim=0,
                index=torch.sort(masked_y_probs, descending=True)[1],
            )
            if torch.sum(masked_sorted_y_trues[:2]).item() > 0:
                top2_counter += 1
        top2: float = top2_counter / len(set(mol_ids_expanded.tolist()))
        return top2


class ValidationLogger(BaseMetrics):
    @classmethod
    def compute_and_log_validation_results(
        cls,
        predictions: dict[int, List[torch.Tensor]],
        descriptions: List[str],
        output_folder: str,
    ) -> None:
        metrics: dict[str, List[float]] = {
            "ROC-AUC": [],
            "PR-AUC": [],
            "F1": [],
            "MCC": [],
            "Precision": [],
            "Recall": [],
            "Top-2 correctness rate": [],
        }

        for fold_id, preds in predictions.items():
            logits = preds[0]
            y_trues = preds[1]
            mol_ids = preds[2]
            atom_ids = preds[3]

            # Compute predicted SoM-probabilities from logits
            y_probs = torch.sigmoid(logits)

            # Expand mol_ids to match the number of atoms
            zero_indices = torch.where(atom_ids == 0)[0]
            num_atoms_per_mol = zero_indices[1:] - zero_indices[:-1]
            num_atoms_per_mol = torch.cat(
                (num_atoms_per_mol, torch.tensor([len(atom_ids) - zero_indices[-1]]))
            )  # Add the last molecule
            mol_ids_expanded = torch.repeat_interleave(mol_ids, num_atoms_per_mol)

            # Expand descriptions to match the number of atoms
            descriptions_expanded = [
                desc
                for desc, count in zip(descriptions, num_atoms_per_mol)
                for _ in range(count)
            ]

            # Compute atom rankings
            rankings = cls.compute_ranking(y_probs, mol_ids_expanded)

            # Compute binary predictions
            y_preds = (y_probs >= THRESHOLD).int()

            # Write results to csv file
            with open(
                os.path.join(output_folder, f"validation_fold{fold_id}.csv"), "w"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    (
                        "mol_id",
                        "atom_id",
                        "y_true",
                        "y_prob",
                        "y_pred",
                        "ranking",
                    )
                )
                for row in zip(
                    descriptions_expanded,
                    atom_ids.tolist(),
                    y_trues.tolist(),
                    [round(y_prob, 4) for y_prob in y_probs.tolist()],
                    y_preds.tolist(),
                    rankings.tolist(),
                ):
                    writer.writerow(row)

            metrics["ROC-AUC"].append(cls.compute_auroc(y_probs, y_trues))
            metrics["PR-AUC"].append(cls.compute_average_precision(y_probs, y_trues))
            metrics["F1"].append(cls.compute_f1(y_preds, y_trues))
            metrics["MCC"].append(cls.compute_mcc(y_preds, y_trues))
            metrics["Precision"].append(cls.compute_precision(y_preds, y_trues))
            metrics["Recall"].append(cls.compute_recall(y_preds, y_trues))
            metrics["Top-2 correctness rate"].append(
                cls.compute_top2(y_probs, y_trues, mol_ids_expanded)
            )

        # Write results to txt file
        with open(os.path.join(output_folder, "validation.txt"), "w") as f:
            for key, value in metrics.items():
                f.write(
                    f"{key}: {round(mean(value), 2)} +/- {round(stdev(value), 2)}\n"
                )


class TestLogger(BaseMetrics):
    @classmethod
    def compute_and_log_test_results(
        cls,
        logits_ensemble: torch.Tensor,
        y_trues: torch.Tensor,
        mol_ids: torch.Tensor,
        atom_ids: torch.Tensor,
        descriptions: List[str],
        output_path: str,
        mode: str,
    ) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compute predicted SoM-probabilities from logits
        y_probs_ensemble = (
            torch.sigmoid(logits_ensemble) + 1e-14
        )  # add epsilon  to avoid issues when computing the log2 of 0 later

        # Compute the averaged predicted SoM-probabilities per atom (i.e., the Bayesian model average)
        y_probs = torch.mean(y_probs_ensemble, dim=0)

        # Compute uncertainties
        u_ales, u_epis, u_tots = cls.compute_uncertainties(y_probs_ensemble)

        # Expand mol_ids to match the number of atoms
        zero_indices = torch.where(atom_ids == 0)[0]
        num_atoms_per_mol = zero_indices[1:] - zero_indices[:-1]
        num_atoms_per_mol = torch.cat(
            (num_atoms_per_mol, torch.tensor([len(atom_ids) - zero_indices[-1]]))
        )  # Add the last molecule
        mol_ids_expanded = torch.repeat_interleave(mol_ids, num_atoms_per_mol)

        # Expand descriptions to match the number of atoms
        descriptions_expanded = [
            desc
            for desc, count in zip(descriptions, num_atoms_per_mol)
            for _ in range(count)
        ]

        # Compute atom rankings
        rankings = cls.compute_ranking(y_probs, mol_ids_expanded)

        # Compute binary predictions
        y_preds = (y_probs >= THRESHOLD).int()

        # Write results to csv file
        row_names = [
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
        if mode == "test":
            row_values = zip(
                descriptions_expanded,
                atom_ids.tolist(),
                y_trues.tolist(),
                [round(y_prob, 4) for y_prob in y_probs.tolist()],
                y_preds.tolist(),
                rankings.tolist(),
                [round(u_ale, 4) for u_ale in u_ales.tolist()],
                [round(u_epi, 4) for u_epi in u_epis.tolist()],
                [round(u_tot, 4) for u_tot in u_tots.tolist()],
            )
        elif mode == "infer":
            row_names.remove("y_true")
            row_values = zip(
                descriptions_expanded,
                atom_ids.tolist(),
                [round(y_prob, 4) for y_prob in y_probs.tolist()],
                y_preds.tolist(),
                rankings.tolist(),
                [round(u_ale, 4) for u_ale in u_ales.tolist()],
                [round(u_epi, 4) for u_epi in u_epis.tolist()],
                [round(u_tot, 4) for u_tot in u_tots.tolist()],
            )
        with open(os.path.join(output_path, "results.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(row_names)
            for row in row_values:
                writer.writerow(row)

        if mode == "test":
            # Initialize empty tensors to hold the metrics for each bootstrap iteration
            aurocs = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            auprs = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            f1s = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            mccs = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            precisions = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            recalls = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            top2s = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")

            mol_ids
            for i in range(NUM_BOOTSTRAPS):
                # Sample molecule IDs with replacement
                sampled_mol_ids = mol_ids[torch.randint(len(mol_ids), (len(mol_ids),))]

                # Create a mask to select atoms of the sampled molecules
                mask = torch.isin(mol_ids_expanded, sampled_mol_ids)

                # Select the values associated with the atoms of the sampled molecules
                mol_id_sample = mol_ids_expanded[mask]
                y_trues_sample = y_trues[mask]
                y_probs_sample = y_probs[mask]
                y_preds_sample = y_preds[mask]

                # Compute metrics
                aurocs[i] = cls.compute_auroc(y_probs_sample, y_trues_sample)
                auprs[i] = cls.compute_average_precision(y_probs_sample, y_trues_sample)
                f1s[i] = cls.compute_f1(y_preds_sample, y_trues_sample)
                mccs[i] = cls.compute_mcc(y_preds_sample, y_trues_sample)
                precisions[i] = cls.compute_precision(y_preds_sample, y_trues_sample)
                recalls[i] = cls.compute_recall(y_preds_sample, y_trues_sample)
                top2s[i] = cls.compute_top2(
                    y_probs_sample, y_trues_sample, mol_id_sample
                )

            # Write results to txt file
            with open(os.path.join(output_path, "results.txt"), "w") as f:
                f.write(
                    f"ROC-AUC: {round(aurocs.mean().item(), 2)} +/- {round(aurocs.std().item(), 2)}\n"
                )
                f.write(
                    f"PR-AUC: {round(auprs.mean().item(), 2)} +/- {round(auprs.std().item(), 2)}\n"
                )
                f.write(
                    f"F1: {round(f1s.mean().item(), 2)} +/- {round(f1s.std().item(), 2)}\n"
                )
                f.write(
                    f"MCC: {round(mccs.mean().item(), 2)} +/- {round(mccs.std().item(), 2)}\n"
                )
                f.write(
                    f"Precision: {round(precisions.mean().item(), 2)} +/- {round(precisions.std().item(), 2)}\n"
                )
                f.write(
                    f"Recall: {round(recalls.mean().item(), 2)} +/- {round(recalls.std().item(), 2)}\n"
                )
                f.write(
                    f"Top-2 correctness rate: {round(top2s.mean().item(), 2)} +/- {round(top2s.std().item(), 2)}\n"
                )

            # Plot ROC curve
            RocCurveDisplay.from_predictions(y_trues, y_probs)
            plt.savefig(str(os.path.join(output_path, "roc.png")), dpi=300)
