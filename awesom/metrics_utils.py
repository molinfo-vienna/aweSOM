import csv
import matplotlib.pyplot as plt
import os
import torch
from torchmetrics import MatthewsCorrCoef, AUROC, ROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from statistics import mean, stdev
from sklearn.metrics import RocCurveDisplay


class BaseMetrics:
    mcc = MatthewsCorrCoef(task="binary")
    auroc = AUROC(task="binary")
    precision = BinaryPrecision()
    recall = BinaryRecall()
    roc = ROC(task="binary")

    @classmethod
    def compute_ranking(cls, y_hat, mol_id):
        ranking = torch.cat(
            [
                torch.argsort(
                    torch.argsort(
                        torch.index_select(
                            y_hat[:, 1], 0, torch.where(mol_id == mid)[0]
                        ),
                        dim=0,
                        descending=True,
                    ),
                    dim=0,
                    descending=False,
                )
                for mid in list(
                    dict.fromkeys(mol_id.tolist())
                )  # This is a somewhat complicated way to get an ordered set, but it works...
            ]
        )
        return ranking


class ValidationMetrics(BaseMetrics):
    @classmethod
    def compute_and_log_validation_metrics(
        cls, predictions: dict, output_folder: str
    ) -> None:
        metrics = {
            "MCC": [],
            "Precision": [],
            "Recall": [],
            "Molecular R-Precision": [],
            "Molecular AUROC": [],
            "Top-2 Correctness Rate": [],
            "Atomic R-Precision": [],
            "Atomic AUROC": [],
        }

        for fold_id, preds in predictions.items():
            y_hat = preds[0][0]
            y = preds[0][1]
            mol_id = preds[0][2]
            atom_id = preds[0][3]

            y_hat_bin = torch.max(y_hat, dim=1).indices
            ranking = cls.compute_ranking(y_hat, mol_id)

            with open(
                os.path.join(output_folder, f"validation_fold{fold_id}.csv"), "w"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    (
                        "probabilities",
                        "ranking",
                        "predicted_labels",
                        "true_labels",
                        "mol_id",
                        "atom_id",
                    )
                )
                for row in zip(
                    y_hat[:, 1].tolist(),
                    ranking.tolist(),
                    y_hat_bin.tolist(),
                    y.tolist(),
                    mol_id.tolist(),
                    atom_id.tolist(),
                ):
                    writer.writerow(row)

            metrics["MCC"].append(cls.mcc(y_hat_bin, y).item())
            metrics["Precision"].append(cls.precision(y_hat_bin, y).item())
            metrics["Recall"].append(cls.recall(y_hat_bin, y).item())
            metrics["Atomic AUROC"].append(cls.auroc(y_hat[:, 1], y).item())

            sorted_y = torch.index_select(
                y, dim=0, index=torch.sort(y_hat[:, 1], descending=True)[1]
            )
            total_num_soms_in_validation_split = torch.sum(y).item()
            metrics["Atomic R-Precision"].append(
                torch.sum(sorted_y[:total_num_soms_in_validation_split]).item()
                / total_num_soms_in_validation_split
            )

            top2_correctness_rate = 0
            per_molecule_aurocs = []
            per_molecule_r_precisions = []
            for id in list(
                dict.fromkeys(mol_id.tolist())
            ):  # This is a somewhat complicated way to get an ordered set, but it works
                mask = torch.where(mol_id == id)[0]
                masked_y = y[mask]
                masked_y_hat = y_hat[mask]
                masked_sorted_y = torch.index_select(
                    masked_y,
                    dim=0,
                    index=torch.sort(masked_y_hat[:, 1], descending=True)[1],
                )
                num_soms_in_current_mol = torch.sum(masked_y).item()
                if torch.sum(masked_sorted_y[:2]).item() > 0:
                    top2_correctness_rate += 1
                per_molecule_aurocs.append(
                    cls.auroc(masked_y_hat[:, 1], masked_y).item()
                )
                per_molecule_r_precisions.append(
                    torch.sum(masked_sorted_y[:num_soms_in_current_mol]).item()
                    / num_soms_in_current_mol
                )
            top2_correctness_rate /= len(set(mol_id.tolist()))
            metrics["Top-2 Correctness Rate"].append(top2_correctness_rate)
            metrics["Molecular AUROC"].append(mean(per_molecule_aurocs))
            metrics["Molecular R-Precision"].append(mean(per_molecule_r_precisions))

        with open(os.path.join(output_folder, "validation.txt"), "w") as f:
            for key, value in metrics.items():
                f.write(
                    f"{key}: {round(mean(value), 2)} +/- {round(stdev(value), 2)}\n"
                )


class TestMetrics(BaseMetrics):
    @classmethod
    def compute_and_log_test_metric(
        cls, predictions: dict, output_folder: str, true_labels: bool
    ) -> None:

        y_hats = []
        for model_id, preds in predictions.items():
            y_hats.append(preds[0][0])
            if model_id == 0:
                y = preds[0][1]
                mol_id = preds[0][2]
                atom_id = preds[0][3]

        y_hats = torch.stack(y_hats, dim=0)
        y_hat_avg = torch.mean(y_hats, dim=0)
        y_hat_bin = torch.max(y_hat_avg, dim=1).indices

        ranking = cls.compute_ranking(y_hat_avg, mol_id)
        with open(os.path.join(output_folder, "results.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    "averaged_probabilities",
                    "ranking",
                    "predicted_labels",
                    "true_labels",
                    "mol_id",
                    "atom_id",
                )
            )
            for row in zip(
                y_hat_avg[:, 1].tolist(),
                ranking.tolist(),
                torch.max(y_hat_avg, dim=1).indices.tolist(),
                y.tolist(),
                mol_id.tolist(),
                atom_id.tolist(),
            ):
                writer.writerow(row)

        if true_labels:
            sorted_y = torch.index_select(
                y, dim=0, index=torch.sort(y_hat_avg[:, 1], descending=True)[1]
            )
            total_num_soms = torch.sum(y).item()
            atom_r_precision = (
                torch.sum(sorted_y[:total_num_soms]).item() / total_num_soms
            )

            top2_correctness_rate = 0
            per_molecule_aurocs = []
            per_molecule_r_precisions = []
            for id in list(
                dict.fromkeys(mol_id.tolist())
            ):  # This is a somewhat complicated way to get an ordered set, but it works
                mask = torch.where(mol_id == id)[0]
                masked_y = y[mask]
                masked_y_hat = y_hat_avg[mask]
                masked_sorted_y = torch.index_select(
                    masked_y,
                    dim=0,
                    index=torch.sort(masked_y_hat[:, 1], descending=True)[1],
                )
                num_soms_in_current_mol = torch.sum(masked_y).item()
                if torch.sum(masked_sorted_y[:2]).item() > 0:
                    top2_correctness_rate += 1
                per_molecule_aurocs.append(
                    cls.auroc(masked_y_hat[:, 1], masked_y).item()
                )
                per_molecule_r_precisions.append(
                    torch.sum(masked_sorted_y[:num_soms_in_current_mol]).item()
                    / num_soms_in_current_mol
                )
            top2_correctness_rate /= len(set(mol_id.tolist()))
            mol_auroc = mean(per_molecule_aurocs)
            mol_r_precision = mean(per_molecule_r_precisions)

            with open(os.path.join(output_folder, "results.txt"), "w") as f:
                f.write(f"MCC: {round(cls.mcc(y_hat_bin, y).item(), 2)}\n")
                f.write(f"Precision: {round(cls.precision(y_hat_bin, y).item(), 2)}\n")
                f.write(f"Recall: {round(cls.recall(y_hat_bin, y).item(), 2)}\n")
                f.write(f"Molecular R-Precision: {round(mol_r_precision, 2)}\n")
                f.write(f"Molecular AUROC:  {round(mol_auroc, 2)}\n")
                f.write(f"Top-2 Correctness Rate: {round(top2_correctness_rate, 2)}\n")
                f.write(f"Atomic R-Precision: {round(atom_r_precision, 2)}\n")
                f.write(
                    f"Atomic AUROC: {round(cls.auroc(y_hat_avg[:, 1], y).item(), 2)}\n"
                )

            RocCurveDisplay.from_predictions(y, y_hat_avg[:, 1])
            plt.savefig(str(os.path.join(output_folder, "roc.png")), dpi=300)
