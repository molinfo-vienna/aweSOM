import csv
import matplotlib.pyplot as plt
import os
import torch
from torchmetrics import MatthewsCorrCoef, AUROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from statistics import mean, stdev
from sklearn.metrics import RocCurveDisplay

NUM_BOOTSTRAPS = 100
THRESHOLD = 0.5

class BaseMetrics:
    @classmethod
    def compute_ranking(cls, y_prob, mol_id):
        ranking = torch.cat(
            [
                torch.argsort(
                    torch.argsort(
                        torch.index_select(y_prob, 0, torch.where(mol_id == mid)[0]),
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
    def compute_uncertainty_score(cls, p):
        return -(abs(2*p - 1)) + 1
    
    @classmethod
    def compute_uncertainties(cls, y_probs, y_probs_avg):
        # # Steffen
        # u_ale = torch.mean(
        #     y_probs * (1 - y_probs), dim=0
        # )  # aleatoric uncertainty
        # u_epi = torch.mean(
        #     y_probs**2 - y_probs_avg**2, dim=0
        # )  # epistemic uncertainty
        # u_tot = (
        #     u_ale + u_epi
        # )

        # Mukhoti et al. and Smith and Gal
        u_tot = cls.compute_shannon_entropy(y_probs_avg)  # entropy of the BMA (a.k.a. predictive entropy)
        u_ale = torch.mean(cls.compute_shannon_entropy(y_probs), dim=0)  # expected shannon entropy of the predictions given the parameters over the posterior distribution
        u_epi = u_tot - u_ale  # mutual information (a.k.a. expected information gain)

        # Map uncertainties to uncertainty scores
        # u_tot = cls.compute_uncertainty_score(y_probs_avg)
        # u_ale = torch.mean(cls.compute_uncertainty_score(y_probs), dim=0)
        # u_epi = u_tot - u_ale

        return u_ale, u_epi, u_tot

    @classmethod
    def scale(cls, x, min, max):
        return (x - min) / (max - min)
    
    @classmethod
    def compute_mcc(cls, y_pred, y_true):
        mcc = MatthewsCorrCoef(task="binary")
        return mcc(y_pred, y_true).item()
    
    @classmethod
    def compute_precision(cls, y_pred, y_true):
        precision = BinaryPrecision()
        return precision(y_pred, y_true).item()
    
    @classmethod
    def compute_recall(cls, y_pred, y_true):
        recall = BinaryRecall()
        return recall(y_pred, y_true).item()
    
    @classmethod
    def compute_auroc(cls, y_prob, y_true):
        auroc = AUROC(task="binary")
        return auroc(y_prob, y_true).item()
    
    @classmethod
    def compute_top2(cls, y_prob, y_true, mol_id):
        top2 = 0
        for id in list(
            dict.fromkeys(mol_id.tolist())
        ):  # This is a somewhat complicated way to get an ordered set (masked_sorted_y), but it works.
            mask = torch.where(mol_id == id)[0]
            masked_y = y_true[mask]
            masked_y_prob = y_prob[mask]
            masked_sorted_y = torch.index_select(
                masked_y,
                dim=0,
                index=torch.sort(masked_y_prob, descending=True)[1],
            )
            if torch.sum(masked_sorted_y[:2]).item() > 0: top2 += 1
        top2 /= len(set(mol_id.tolist()))
        return top2
    

class ValidationLogger(BaseMetrics):
    @classmethod
    def compute_and_log_validation_results(
        cls, predictions: dict, output_folder: str
    ) -> None:
        metrics = {
            "MCC": [],
            "Precision": [],
            "Recall": [],
            "AUROC": [],
            "Top-2 correctness rate": [],
        }

        for fold_id, preds in predictions.items():
            logits = preds[0][0]
            y_true = preds[0][1]
            mol_id = preds[0][2]
            atom_id = preds[0][3]

            y_prob = torch.sigmoid(logits)
            ranking = cls.compute_ranking(y_prob, mol_id)
            y_pred = (y_prob >= THRESHOLD).int()

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
                    mol_id.tolist(),
                    atom_id.tolist(),
                    y_true.tolist(),
                    y_prob.tolist(),
                    y_pred.tolist(),
                    ranking.tolist(),
                ):
                    writer.writerow(row)

            metrics["MCC"].append(cls.compute_mcc(y_pred, y_true))
            metrics["Precision"].append(cls.compute_precision(y_pred, y_true))
            metrics["Recall"].append(cls.compute_recall(y_pred, y_true))
            metrics["AUROC"].append(cls.compute_auroc(y_prob, y_true))
            metrics["Top-2 correctness rate"].append(cls.compute_top2(y_prob, y_true, mol_id))

        with open(os.path.join(output_folder, "validation.txt"), "w") as f:
            for key, value in metrics.items():
                f.write(
                    f"{key}: {round(mean(value), 4)} +/- {round(stdev(value), 4)}\n"
                )

class TestLogger(BaseMetrics):
    @classmethod
    def compute_and_log_test_results(
        cls,
        mol_id: torch.Tensor,
        atom_id: torch.Tensor,
        y_true: torch.Tensor,
        logits: torch.Tensor,
        output_path: str, 
        mode: str,
    ) -> None:

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Compute predicted SoM-probabilities from logits
        y_probs = (torch.sigmoid(logits) + 1e-14)  # add epsilon  to avoid issues when computing the log2 of 0 later

        # Compute the averaged predicted SoM-probability for the ensemble
        y_prob= torch.mean(y_probs, dim=0)

        # Compute uncertainties
        u_ale, u_epi, u_tot = cls.compute_uncertainties(y_probs, y_prob)

        # Compute atom rankings
        ranking = cls.compute_ranking(y_prob, mol_id)

        # Compute binary predictions
        y_pred = (y_prob >= THRESHOLD).int()

        # Write results to csv file
        with open(os.path.join(output_path, "results.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    "mol_id",
                    "atom_id",
                    "y_true",
                    "y_prob",
                    "y_pred",
                    "ranking",
                    "u_ale",
                    "u_epi",
                    "u_tot",
                )
            )
            for row in zip(
                mol_id.tolist(),
                atom_id.tolist(),
                y_true.tolist(),
                y_prob.tolist(),
                y_pred.tolist(),
                ranking.tolist(),
                u_ale.tolist(),
                u_epi.tolist(),
                u_tot.tolist(),
            ):
                writer.writerow(row)

        if mode == "test":
            # Initialize empty tensors to hold the metrics for each bootstrap iteration
            mccs = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            precisions = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            recalls = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            aurocs = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")
            top2s = torch.empty(NUM_BOOTSTRAPS, dtype=torch.float32, device="cpu")

            num_mols = len(torch.unique(mol_id))
            num_sampled_mols = int(0.9 * num_mols)
            for i in range(100):
                # Get a random 90% of the data (by molecular ID so that a substrate is not split across different bootstrap iterations)
                sampled_mol_ids = torch.randperm(num_mols)[:num_sampled_mols]
                mask = torch.zeros_like(mol_id, dtype=torch.bool)
                for id in sampled_mol_ids:
                    mask = mask | (mol_id == id)
                    mol_id_sample = mol_id[mask]
                    y_true_sample = y_true[mask]
                    y_prob_sample = y_prob[mask]
                    y_pred_sample = y_pred[mask]
                    
                # Compute metrics
                mccs[i] = cls.compute_mcc(y_pred_sample, y_true_sample)
                precisions[i] = cls.compute_precision(y_pred_sample, y_true_sample)
                recalls[i] = cls.compute_recall(y_pred_sample, y_true_sample)
                aurocs[i] = cls.compute_auroc(y_prob_sample, y_true_sample)
                top2s[i] = cls.compute_top2(y_prob_sample, y_true_sample, mol_id_sample)

            # Write results to txt file
            with open(os.path.join(output_path, "results.txt"), "w") as f:
                f.write(f"MCC: {round(mccs.mean().item(), 4)} +/- {round(mccs.std().item(), 4)}\n")
                f.write(f"Precision: {round(precisions.mean().item(), 4)} +/- {round(precisions.std().item(), 4)}\n")
                f.write(f"Recall: {round(recalls.mean().item(), 4)} +/- {round(recalls.std().item(), 4)}\n")
                f.write(f"AUROC: {round(aurocs.mean().item(), 4)} +/- {round(aurocs.std().item(), 4)}\n")
                f.write(f"Top-2 correctness rate: {round(top2s.mean().item(), 4)} +/- {round(top2s.std().item(), 4)}\n")

            # Plot ROC curve  
            RocCurveDisplay.from_predictions(y_true, y_prob)
            plt.savefig(str(os.path.join(output_path, "roc.png")), dpi=300)
