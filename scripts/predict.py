import argparse
import csv
import os
import torch

from lightning import Trainer
from torchmetrics import MatthewsCorrCoef, AUROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torch_geometric.loader import DataLoader

from awesom.models import (
    GATv2,
    GIN,
    GINE,
    NN,
)
from awesom.dataset import LabeledData, UnlabeledData
from awesom.utils import (
    seed_everything,
)

THRESHOLD = 0.2


def run_predict():
    if not os.path.exists(args.outputFolder):
        os.makedirs(args.outputFolder)

    if args.trueLabels == "False":
        data = UnlabeledData(args.inputFolder)
    else:
        data = LabeledData(args.inputFolder)

    print(f"Number of molecules: {len(data)}")

    best_model_paths = []
    with open(os.path.join(args.logFolder, "best_model_paths.txt"), "r") as f:
        best_model_paths = f.read().splitlines()

    y_hat_avg = None
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
        elif args.model == "NN":
            model = NN.load_from_checkpoint(path)

        trainer = Trainer(accelerator="auto", logger=False)
        out = trainer.predict(model, DataLoader(data, batch_size=len(data)))
        y_hat = out[0][0]
        if i == 0:
            y_hat_avg = y_hat
            y = out[0][1]
            mol_id = out[0][2]
            atom_id = out[0][3]
        else:
            y_hat_avg = torch.vstack((y_hat_avg, y_hat))

    y_hat_avg = torch.mean(y_hat_avg, dim=0)

    with open(os.path.join(args.outputFolder, "results.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(("predicted_labels", "true_labels", "mol_id", "atom_id"))
        for row in zip(
            y_hat_avg.tolist(),
            y.tolist(),
            mol_id.tolist(),
            atom_id.tolist(),
        ):
            writer.writerow(row)

    if args.trueLabels == "True":
        mcc = MatthewsCorrCoef(task="binary", threshold=THRESHOLD)
        auroc = AUROC(task="binary")
        precision = BinaryPrecision(threshold=THRESHOLD)
        recall = BinaryRecall(threshold=THRESHOLD)

        with open(os.path.join(args.outputFolder, "results.txt"), "w") as f:
            f.write(f"MCC: {mcc(y_hat_avg, y)}\n")
            f.write(f"AUROC: {auroc(y_hat_avg, y)}\n")
            f.write(f"Precision: {precision(y_hat_avg, y)}\n")
            f.write(f"Recall: {recall(y_hat_avg, y)}\n")

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
        help="The desired model architecture. Choose between 'GATv2', 'GIN', 'GINE' and 'NN'.",
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

    seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    run_predict()
