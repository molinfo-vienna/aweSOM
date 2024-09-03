import argparse
import os
import torch
import yaml

from datetime import datetime
from lightning import Trainer
from pathlib import Path
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.dataset import LabeledData
from awesom.lightning_modules import GNN
from awesom.metrics_utils import TestLogger

NUM_MONTE_CARLO_SAMPLES = 50


def main():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # Load data
    data = LabeledData(root=args.inputPath, transform=T.ToUndirected())
    print(f"Number of molecules: {len(data)}")

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
    hyperparams["hyperparams"]["mode"] = "mcdropout"

    model = GNN(
        params=hyperparams["params"],
        hyperparams=hyperparams["hyperparams"],
        architecture=hyperparams["architecture"],
    )

    model = GNN.load_from_checkpoint(checkpoints_file)

    # Predict SoMs
    mol_id_mcsampled = torch.empty(len(data.x), dtype=torch.int64, device="cpu")  # 1D tensor
    atom_id_mcsampled = torch.empty(len(data.x), dtype=torch.int64, device="cpu")  # 1D tensor
    y_true_mcsampled = torch.empty(len(data.x), dtype=torch.int64, device="cpu")  # 1D tensor
    logits_mcsampled = torch.empty((NUM_MONTE_CARLO_SAMPLES, len(data.x)), dtype=torch.float32, device="cpu")  # 2D tensor
    
    for i in range(NUM_MONTE_CARLO_SAMPLES):
        # Initialize trainer
        trainer = Trainer(accelerator="auto", logger=False)

        # Make predictions
        logits, y_true, mol_id, atom_id = trainer.predict(
            model=model, dataloaders=DataLoader(data, batch_size=len(data.x))
        )[0]

        
        if i == 0:
            mol_id_mcsampled[:] = mol_id
            atom_id_mcsampled[:] = atom_id
            y_true_mcsampled[:] = y_true
        logits_mcsampled[i, :] = logits

    TestLogger.compute_and_log_test_results(
        mol_id_mcsampled,
        atom_id_mcsampled,
        y_true_mcsampled,
        logits_mcsampled,      
        args.outputPath,
        args.mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predicting SoMs for labeled (test) and unlabeled (infer) data.")

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
        "-m",
        dest="mode",
        type=str,
        required=True,
        help="The mode of the model. Must be either 'test' or 'infer'.",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    main()
    print("Finished in:")
    print(datetime.now() - start_time)
