import argparse
import os
import torch
import yaml

from lightning import Trainer, seed_everything
from pathlib import Path
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom import LabeledData, UnlabeledData, GNN, TestMetrics


def run_predict():
    seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)

    if args.test:
        data = LabeledData(root=args.inputPath, transform=T.ToUndirected())
    else:
        data = UnlabeledData(root=args.inputPath, transform=T.ToUndirected())

    print(f"Number of molecules: {len(data)}")

    best_model_paths = []
    with open(Path(args.checkpointsPath, "best_model_paths.txt"), "r") as f:
        best_model_paths = f.read().splitlines()

    predictions = {}
    for i, path in enumerate(best_model_paths):
        # get checkpoints
        for file in os.listdir(Path(path, "checkpoints")):
            if file.endswith(".ckpt"):
                checkpoint_path = Path(Path(path, "checkpoints"), file)

        model = GNN.load_from_checkpoint(checkpoint_path)

        trainer = Trainer(accelerator="auto", logger=False)
        predictions[i] = trainer.predict(
            model=model, dataloaders=DataLoader(data, batch_size=len(data))
        )

    TestMetrics.compute_and_log_test_metrics(predictions, args.outputPath, args.test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predicting SoMs for unseen data.")

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
        help="Whether to performance inference (False, default value) or testing (True). \
                If set to true, the script assumes that true labels are provided \
                and computes the classification metrics (MCC, precision, recall, top2 correctness rate, \
                atomic and molecular AUROCs, and atomic and molecular R-precisions).",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    run_predict()
