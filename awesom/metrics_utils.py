import csv
import matplotlib.pyplot as plt
import os
import torch
from torchmetrics import MatthewsCorrCoef, AUROC, ROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from statistics import mean, stdev
from sklearn.metrics import RocCurveDisplay

THRESHOLD = 0.3


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
                        torch.index_select(y_hat, 0, torch.where(mol_id == mid)[0]),
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

    @classmethod
    def compute_shannon_entropy(cls, p):
        return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))

    @classmethod
    def scale(cls, x, min, max):
        return (x - min) / (max - min)


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
            logits = preds[0][0]
            stddevs = preds[0][1]
            y = preds[0][2]
            mol_id = preds[0][3]
            atom_id = preds[0][4]

            y_hat = torch.sigmoid(logits)
            ranking = cls.compute_ranking(y_hat, mol_id)
            y_hat_bin = (y_hat >= THRESHOLD).int()

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
                    y_hat.tolist(),
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
            metrics["Atomic AUROC"].append(cls.auroc(y_hat, y).item())

            sorted_y = torch.index_select(
                y, dim=0, index=torch.sort(y_hat, descending=True)[1]
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
                    index=torch.sort(masked_y_hat, descending=True)[1],
                )
                num_soms_in_current_mol = torch.sum(masked_y).item()
                if torch.sum(masked_sorted_y[:2]).item() > 0:
                    top2_correctness_rate += 1
                per_molecule_aurocs.append(cls.auroc(masked_y_hat, masked_y).item())
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
                    f"{key}: {round(mean(value), 4)} +/- {round(stdev(value), 4)}\n"
                )


class TestMetrics(BaseMetrics):
    @classmethod
    def compute_and_log_test_metrics(
        cls, logits, stddevs, y, mol_id, atom_id, output_folder: str, true_labels: bool
    ) -> None:
        y_hats = (
            torch.sigmoid(logits) + 1e-20
        )  # add epsilon  to avoid issues when computing the log2 of 0 later

        y_hats_avg = torch.mean(
            y_hats, dim=0
        )  # mean predicted probabilities of ensemble (predicted SoM probabilities)

        # # Steffen
        # sigma_ale = torch.mean(
        #     y_hats * (1 - y_hats), dim=0
        # )  # aleatoric uncertainty (this one also works quite well, and is very similar to the Mucsanyi, Kirchhof and Oh method)
        # sigma_epi = torch.mean(
        #     y_hats**2 - y_hats_avg**2, dim=0
        # )  # epistemic uncertainty (this one is also unsatisafactory, and interestingly also very similar to the Mucsanyi, Kirchhof and Oh method, second variant)
        # sigma_tot = (
        #     sigma_ale + sigma_epi
        # )  # total uncertainty as the sum of aleatoric and epistemic uncertainty

        # # Mine
        # sigma_ale = torch.mean(cls.compute_shannon_entropy(y_hats), dim=0)  # expected value of the shannon entropy of the sampled probabilities
        # sigma_epi = abs(torch.std(torch.log(y_hats), dim=0) / torch.mean(torch.log(y_hats), dim=0)) # coefficient of variation 
        # sigma_tot = cls.compute_shannon_entropy(y_hats_avg) # entropy of the BMA (see Gustafsson et al. 2020)

        # Mukhoti et al. and Smith and Gal
        sigma_tot = cls.compute_shannon_entropy(y_hats_avg)  # entropy of the BMA (a.k.a. predictive entropy)
        sigma_ale = torch.mean(cls.compute_shannon_entropy(y_hats), dim=0)  # expected shannon entropy of the predictions given the parameters over the posterior distribution
        sigma_epi = sigma_tot - sigma_ale  # mutual information (a.k.a. expected information gain)

        ranking = cls.compute_ranking(y_hats_avg, mol_id)

        y_hat_bin = (
            y_hats_avg >= THRESHOLD
        ).int()

        with open(os.path.join(output_folder, "results.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    "averaged_probabilities",
                    "aleatoric_uncertainty",
                    "epistemic_uncertainty",
                    "total_uncertainty",
                    "ranking",
                    "predicted_binary_labels",
                    "true_labels",
                    "mol_id",
                    "atom_id",
                )
            )
            for row in zip(
                y_hats_avg.tolist(),
                sigma_ale.tolist(),
                sigma_epi.tolist(),
                sigma_tot.tolist(),
                ranking.tolist(),
                y_hat_bin.tolist(),
                y.tolist(),
                mol_id.tolist(),
                atom_id.tolist(),
            ):
                writer.writerow(row)

        if true_labels:
            sorted_y = torch.index_select(
                y, dim=0, index=torch.sort(y_hats_avg, descending=True)[1]
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
                masked_y_hat = y_hats_avg[mask]
                masked_sorted_y = torch.index_select(
                    masked_y,
                    dim=0,
                    index=torch.sort(masked_y_hat, descending=True)[1],
                )
                num_soms_in_current_mol = torch.sum(masked_y).item()
                if torch.sum(masked_sorted_y[:2]).item() > 0:
                    top2_correctness_rate += 1
                per_molecule_aurocs.append(cls.auroc(masked_y_hat, masked_y).item())
                per_molecule_r_precisions.append(
                    torch.sum(masked_sorted_y[:num_soms_in_current_mol]).item()
                    / num_soms_in_current_mol
                )
            top2_correctness_rate /= len(set(mol_id.tolist()))
            mol_auroc = mean(per_molecule_aurocs)
            mol_r_precision = mean(per_molecule_r_precisions)

            with open(os.path.join(output_folder, "results.txt"), "w") as f:
                f.write(f"MCC: {round(cls.mcc(y_hat_bin, y).item(), 4)}\n")
                f.write(f"Precision: {round(cls.precision(y_hat_bin, y).item(), 4)}\n")
                f.write(f"Recall: {round(cls.recall(y_hat_bin, y).item(), 4)}\n")
                f.write(f"Molecular R-Precision: {round(mol_r_precision, 4)}\n")
                f.write(f"Molecular AUROC:  {round(mol_auroc, 4)}\n")
                f.write(f"Top-2 Correctness Rate: {round(top2_correctness_rate, 4)}\n")
                f.write(f"Atomic R-Precision: {round(atom_r_precision, 4)}\n")
                f.write(f"Atomic AUROC: {round(cls.auroc(y_hats_avg, y).item(), 4)}\n")

            RocCurveDisplay.from_predictions(y, y_hats_avg)
            plt.savefig(str(os.path.join(output_folder, "roc.png")), dpi=300)
