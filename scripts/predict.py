import argparse
import csv
import matplotlib.pyplot as plt
import os
import torch
import yaml

from lightning import Trainer, seed_everything
from torchmetrics import MatthewsCorrCoef, AUROC
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torch_geometric.loader import DataLoader
from sklearn.metrics import RocCurveDisplay
from statistics import mean

from awesom.dataset import LabeledData, UnlabeledData
from awesom.metrics_utils import compute_ranking
from awesom.models import GNN


mcc = MatthewsCorrCoef(task="binary")
auroc = AUROC(task="binary")
precision = BinaryPrecision()
recall = BinaryRecall()


def run_predict():
    seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    if not os.path.exists(args.outputFolder):
        os.makedirs(args.outputFolder)

    if args.trueLabels == "False":
        data = UnlabeledData(root=args.inputFolder, transform=T.ToUndirected())
    else:
        data = LabeledData(root=args.inputFolder, transform=T.ToUndirected())

    print(f"Number of molecules: {len(data)}")

    best_model_paths = []
    with open(os.path.join(args.modelFolder, "best_model_paths.txt"), "r") as f:
        best_model_paths = f.read().splitlines()

    y_hats = []
    for i, path in enumerate(best_model_paths):

        # get checkpoints
        for file in os.listdir(os.path.join(path, "checkpoints")):
            if file.endswith(".ckpt"):
                checkpoint_path = os.path.join(os.path.join(path, "checkpoints"), file)
        # get best threshold from haparams.yaml
        for file in os.listdir(path):
            if file.endswith(".yaml"):
                hparams_file = os.path.join(path, file)
        with open(hparams_file, 'r') as f:
            best_threshold = yaml.safe_load(f)["threshold"]

        model = GNN.load_from_checkpoint(checkpoint_path, threshold=best_threshold)

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

    # Use this to compute binary predictions when using class weights
    y_hat_bin = torch.max(y_hat_avg, dim=1).indices

    # Use this to compute binary predictions when using threshold moving
    # best_threshold = model.hparams.threshold
    # y_hat_bin = (y_hat_avg[:,1] > best_threshold).to(int)

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

        # Compute weird metrics from Porokhin's GNN-SOM paper for comparison's sake...
        sorted_y = torch.index_select(y, dim=0, index=torch.sort(y_hat[:, 1], descending=True)[1])
        total_num_soms = torch.sum(y).item()
        atom_r_precision = torch.sum(sorted_y[:total_num_soms]).item() / total_num_soms

        top2_correctness_rate = 0
        per_molecule_aurocs = []
        per_molecule_r_precisions = []
        for id in list(dict.fromkeys(mol_id.tolist())):  # This is a somewhat complicated way to get an ordered set, but it works
            mask = torch.where(mol_id == id)[0]
            masked_y = y[mask]
            masked_y_hat = y_hat[mask]
            masked_sorted_y = torch.index_select(masked_y, dim=0, index=torch.sort(masked_y_hat[:, 1], descending=True)[1])
            num_soms_in_current_mol = torch.sum(masked_y).item()
            if torch.sum(masked_sorted_y[:2]).item() > 0:
                top2_correctness_rate += 1 
            per_molecule_aurocs.append(auroc(masked_y_hat[:,1], masked_y).item())
            per_molecule_r_precisions.append(torch.sum(masked_sorted_y[:num_soms_in_current_mol]).item() / num_soms_in_current_mol)
        top2_correctness_rate /= len(set(mol_id.tolist()))
        mol_auroc = mean(per_molecule_aurocs)
        mol_r_precision = mean(per_molecule_r_precisions)

        with open(os.path.join(args.outputFolder, "results.txt"), "w") as f:
            f.write(f"MCC: {round(mcc(y_hat_bin, y).item(), 2)}\n")
            f.write(f"Precision: {round(precision(y_hat_bin, y).item(), 2)}\n")
            f.write(f"Recall: {round(recall(y_hat_bin, y).item(), 2)}\n")
            f.write(f"Molecular R-Precision: {round(mol_r_precision, 2)}\n")
            f.write(f"Molecular AUROC:  {round(mol_auroc, 2)}\n")
            f.write(f"Top-2 Correctness Rate: {round(top2_correctness_rate, 2)}\n")
            f.write(f"Atomic R-Precision: {round(atom_r_precision, 2)}\n")
            f.write(f"Atomic AUROC: {round(auroc(y_hat_avg[:, 1], y).item(), 2)}\n")

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
        "-m",
        "--modelFolder",
        type=str,
        required=True,
        help="The folder where the model's checkpoints are stored.",
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
