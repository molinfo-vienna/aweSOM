import argparse
import os
import torch
import yaml

from datetime import datetime
from lightning import Trainer, seed_everything
from multiprocessing import cpu_count
from pathlib import Path
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.dataset import LabeledData, UnlabeledData   
from awesom.lightning_modules import GNN
from awesom.metrics_utils import TestLogger


def main():
    torch.manual_seed(42)
    seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # Load data
    if args.mode == "test":
        data = LabeledData(root=args.inputPath, transform=T.ToUndirected())
    elif args.mode == "infer":
        data = UnlabeledData(root=args.inputPath, transform=T.ToUndirected())
    else:
        raise ValueError("The mode must be either 'test' or 'infer'.")
    print(f"Number of molecules: {len(data)}")

    # Load ensemble's checkpoints
    checkpoints_path = Path(args.checkpointsPath, "lightning_logs")
    version_paths = [
        Path(checkpoints_path, f"version_{i}")
        for i, _ in enumerate(os.listdir(checkpoints_path))
    ]

    # Initialize empty prediction tensors
    mol_id_ensemble = torch.empty(
        len(data.x), dtype=torch.int64, device="cpu"
    )  # 1D tensor
    atom_id_ensemble = torch.empty(
        len(data.x), dtype=torch.int64, device="cpu"
    )  # 1D tensor
    y_true_ensemble = torch.empty(
        len(data.x), dtype=torch.int64, device="cpu"
    )  # 1D tensor
    logits_ensemble = torch.empty(
        (len(version_paths), len(data.x)), dtype=torch.float32, device="cpu"
    )  # 2D tensor

    # Predict SoMs for each model in the ensemble
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
        logits, y_true, mol_id, atom_id = trainer.predict(
            model=model, dataloaders=DataLoader(data, 
                                                batch_size=len(data),
                                                shuffle=False,)
        )[0]

        if i == 0:
            mol_id_ensemble[:] = mol_id
            atom_id_ensemble[:] = atom_id
            y_true_ensemble[:] = y_true
        logits_ensemble[i, :] = logits

    # Compute and log test results
    TestLogger.compute_and_log_test_results(
        mol_id_ensemble,
        atom_id_ensemble,
        y_true_ensemble,
        logits_ensemble,
        args.outputPath,
        args.mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Predicting SoMs for labeled (test) and unlabeled (infer) data."
    )

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
