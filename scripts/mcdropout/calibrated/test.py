import argparse
import os
import torch
import yaml

from datetime import datetime
from lightning import Trainer
from pathlib import Path
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.dataset import LabeledData, UnlabeledData
from awesom.lightning_modules import EnsembleGNN
from awesom.metrics_utils import TestMetrics


def main():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # Load model
    checkpoints_path = Path(
        Path(Path(args.checkpointsPath, "lightning_logs"), "version_0"), "checkpoints"
    )
    checkpoints_file = Path(
        checkpoints_path,
        [file for file in os.listdir(checkpoints_path) if file.endswith(".ckpt")][0],
    )

    hyperparams_path = Path(
        Path(Path(args.checkpointsPath, "lightning_logs"), "version_0"), "hparams.yaml"
    )
    hyperparams = yaml.safe_load(hyperparams_path.read_text())

    model = EnsembleGNN(
        params=hyperparams["params"],
        hyperparams=hyperparams["hyperparams"],
        architecture=hyperparams["architecture"],
    )

    model = EnsembleGNN.load_from_checkpoint(checkpoints_file)

    # Load data
    if args.test:
        data = LabeledData(root=args.inputPath, transform=T.ToUndirected())
    else:
        data = UnlabeledData(root=args.inputPath, transform=T.ToUndirected())

    print(f"Number of molecules: {len(data)}")

    # Initialize trainer
    trainer = Trainer(accelerator="auto", logger=False)

    # Make predictions
    predictions = trainer.predict(
        model=model, dataloaders=DataLoader(data, batch_size=len(data))
    )[0]
    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)

    TestMetrics.compute_and_log_test_metrics(
        predictions[0].to("cpu"),
        predictions[1].to("cpu"),
        predictions[2].to("cpu"),
        predictions[3].to("cpu"),
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
