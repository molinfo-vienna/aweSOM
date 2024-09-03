import argparse
import os
import torch
import yaml

from datetime import datetime
from lightning import Trainer, seed_everything
from pathlib import Path
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.dataset import LabeledData, UnlabeledData
from awesom.lightning_modules import GNN
from awesom.metrics_utils import TestMetrics


def main():
    torch.manual_seed(42)
    seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # Load data
    if args.test:
        data = LabeledData(root=args.inputPath, transform=T.ToUndirected())
    else:
        data = UnlabeledData(root=args.inputPath, transform=T.ToUndirected())

    print(f"Number of molecules: {len(data)}")

    checkpoints_path = Path(args.checkpointsPath, "lightning_logs")
    version_paths = [
        Path(checkpoints_path, f"version_{i}")
        for i, _ in enumerate(os.listdir(checkpoints_path))
    ]

    logits_ensemble = torch.empty(
        len(version_paths), len(data.x), dtype=torch.float32, device="cpu"
    )
    y_ensemble = torch.empty(len(data.x), dtype=torch.int64, device="cpu")
    mol_id_ensemble = torch.empty(len(data.x), dtype=torch.int64, device="cpu")
    atom_id_ensemble = torch.empty(len(data.x), dtype=torch.int64, device="cpu")

    for i, version_path in enumerate(version_paths):
        checkpoint_path = [
            Path(Path(version_path, "checkpoints"), file)
            for file in os.listdir(Path(version_path, "checkpoints"))
            if file.endswith(".ckpt")
        ][0]
        hyperparams = yaml.safe_load(Path(version_path, "hparams.yaml").read_text())
        hyperparams["hyperparams"]["mode"] = "ensemble"

        # Load model
        model = GNN(
            params=hyperparams["params"],
            hyperparams=hyperparams["hyperparams"],
            architecture=hyperparams["architecture"],
        )
        model = GNN.load_from_checkpoint(checkpoint_path)

        # Predict SoMs
        trainer = Trainer(accelerator="auto", logger=False)
        logits, y, mol_id, atom_id = trainer.predict(
            model=model, dataloaders=DataLoader(data, batch_size=len(data))
        )[0]

        logits_ensemble[i, :] = logits
        if i == 0:
            y_ensemble[:] = y
            mol_id_ensemble[:] = mol_id
            atom_id_ensemble[:] = atom_id

    # Compute and log test metrics
    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)

    TestMetrics.compute_and_log_test_metrics(
        logits_ensemble,
        y_ensemble,
        mol_id_ensemble,
        atom_id_ensemble,
        args.outputPath,
        args.test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predicting SoMs for new data.")

    parser.add_argument(
        "-i",
        dest="inputPath",
        type=str,
        required=True,
        help="The path to the input data.",
    )
    parser.add_argument(
        "-c",
        dest="checkpointsPath",
        type=str,
        required=True,
        help="The path to the model's checkpoints.",
    )
    parser.add_argument(
        "-o",
        dest="outputPath",
        type=str,
        required=True,
        help="The desired output's location.",
    )
    parser.add_argument(
        "-t",
        dest="test",
        required=False,
        help="  Whether to perform inference on non-labeled data (False, default value) \
                or testing on labeled data (True). \
                If set to true, the script assumes that true labels are provided \
                and computes the classification metrics (MCC, precision, recall, top2 correctness rate, \
                atomic and molecular AUROCs, and atomic and molecular R-precisions).",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    start_time = datetime.now()
    main()
    print("Finished in:")
    print(datetime.now() - start_time)
