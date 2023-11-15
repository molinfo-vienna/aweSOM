import argparse
import csv
import matplotlib.pyplot as plt
import os
import torch

from lightning import Trainer, seed_everything
from torchmetrics import MatthewsCorrCoef, AUROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torch_geometric.loader import DataLoader
from sklearn.metrics import RocCurveDisplay

from awesom.dataset import LabeledData, UnlabeledData
from awesom.metrics_utils import compute_ranking
from awesom.models import (
    GATv2,
    GIN,
    GINE,
    GINP,
    MF,
    Cheb,
)


def run_predict():
    seed_everything(42, workers=True)

    if not os.path.exists(args.outputFolder):
        os.makedirs(args.outputFolder)

    if args.trueLabels == "False":
        data = UnlabeledData(root=args.inputFolder)
    else:
        data = LabeledData(root=args.inputFolder)

    print(f"Number of molecules: {len(data)}")

    best_model_paths = []
    with open(os.path.join(args.logFolder, "best_model_paths.txt"), "r") as f:
        best_model_paths = f.read().splitlines()

    y_hats = []
    for i, path in enumerate(best_model_paths):
        for file in os.listdir(path):
            if file.endswith(".ckpt"):
                path = os.path.join(path, file)

        if args.model == "GATv2":
            model = GATv2.load_from_checkpoint(path)
        elif args.model == "GIN":
            model = GIN.load_from_checkpoint(path)
        elif args.model == "GINE":
            model = GINE.load_from_checkpoint(path)
        elif args.model == "GINP":
            model = GINP.load_from_checkpoint(path)
        elif args.model == "MF":
            model = MF.load_from_checkpoint(path)
        elif args.model == "Cheb":
            model = Cheb.load_from_checkpoint(path)

        trainer = Trainer(accelerator="auto", logger=False)
        out = trainer.predict(
            model=model, dataloaders=DataLoader(data, batch_size=len(data))
        )
        y_hat = out[0][0]
        y_hats.append(y_hat)
        if i == 0:
            y = out[0][1]
            mol_id = out[0][2]
            atom_id = out[0][3]

    y_hats = torch.stack(y_hats, dim=0)
    y_hat_avg = torch.mean(y_hats, dim=0)
    y_hat_bin = torch.max(y_hat_avg, dim=1).indices

    # y_hats_bin = torch.stack([torch.max(y_hat, dim=1).indices for y_hat in y_hats], dim=0)
    # y_hat_voted = torch.sum(y_hats_bin, dim=0)  >= y_hats_bin.shape[0]/2
    # y_hat_any = torch.any(y_hats_bin, dim=0)

    ranking = compute_ranking(y_hat_avg, mol_id)

    with open(os.path.join(args.outputFolder, "results.csv"), "w") as f:
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

    if args.trueLabels == "True":
        mcc = MatthewsCorrCoef(task="binary")
        auroc = AUROC(task="multiclass", num_classes=2)
        precision = BinaryPrecision(threshold=0.5)
        recall = BinaryRecall(threshold=0.5)

        with open(os.path.join(args.outputFolder, "results.txt"), "w") as f:
            f.write(f"AUROC: {round(auroc(y_hat_avg, y).item(), 2)}\n")
            f.write(f"MCC: {round(mcc(y_hat_bin, y).item(), 2)}\n")
            f.write(f"Precision: {round(precision(y_hat_bin, y).item(), 2)}\n")
            f.write(f"Recall: {round(recall(y_hat_bin, y).item(), 2)}\n")

        RocCurveDisplay.from_predictions(y, y_hat_avg[:, 1])
        plt.savefig(str(os.path.join(args.outputFolder, "roc.png")), dpi=300)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predicting SoMs for unseen data.")

    parser.add_argument(
        "-i",
        "--inputFolder",
        type=str,
        required=True,
        help="The folder where the input data is stored.",
    )
    parser.add_argument(
        "-l",
        "--logFolder",
        type=str,
        required=True,
        help="The folder where the model's checkpoints are stored.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The desired model architecture.",
    )
    parser.add_argument(
        "-t",
        "--trueLabels",
        type=str,
        required=True,
        default="False",
        help="Whether or not your input data has true labels. If set to true, predict.py will compute classification metrics MCC, AUROC, precision and recall.",
    )
    parser.add_argument(
        "-o",
        "--outputFolder",
        type=str,
        required=True,
        help="The folder where the output will be written.",
    )

    args = parser.parse_args()
    run_predict()
