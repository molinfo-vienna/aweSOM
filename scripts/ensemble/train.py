import argparse
import random
import torch
import yaml

from datetime import datetime
from lightning import seed_everything as lightning_seed_everything
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom.dataset import SOM
from awesom.lightning_modules import GNN

BATCH_SIZE = 32
ENSEMBLE_SIZE = 50


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    random_seeds = random.sample(range(0, 1000), ENSEMBLE_SIZE)

    # Load data
    data = SOM(root=args.inputPath, transform=T.ToUndirected())
    data_params = dict(
        num_node_features=data.num_node_features,
        num_edge_features=data.num_edge_features,
        # num_mol_features=data.mol_x.shape[1],
    )

    for seed in random_seeds:
        torch.manual_seed(42)
        lightning_seed_everything(seed)
        geometric_seed_everything(seed)

        train_data, val_data = train_test_split(data, test_size=0.1, random_state=seed)

        print(f"Number of training instances: {len(train_data)}")
        print(f"Number of validation instances: {len(val_data)}")

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_count(), persistent_workers=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_count(), persistent_workers=True)

        # Load model
        hyperparams = yaml.safe_load(
            Path(args.hparamsYamlPath, "best_hparams.yaml").read_text()
        )
        hyperparams["mode"] = "ensemble"
        model = GNN(
            params=data_params,
            hyperparams=hyperparams,
            architecture=args.model,
        )

        # Initialize trainer
        tbl = TensorBoardLogger(
            save_dir=Path(args.outputPath),
            default_hp_metric=False,
        )

        callbacks = [
            EarlyStopping(monitor="val/loss", mode="min", min_delta=0, patience=20),
            ModelCheckpoint(monitor="val/loss", mode="min"),
        ]

        trainer = Trainer(
            accelerator="auto",
            max_epochs=args.epochs,
            logger=tbl,
            log_every_n_steps=1,
            callbacks=callbacks,
        )

        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    with open(Path(args.outputPath, "random_seeds.txt"), "w+") as f:
        for seed in random_seeds:
            f.write(f"{seed}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training the deep ensemble model.")

    parser.add_argument(
        "-i",
        dest="inputPath",
        type=str,
        required=True,
        help="The path to the input data.",
    )
    parser.add_argument(
        "-c",
        dest="hparamsYamlPath",
        type=str,
        required=True,
        help="The path to the yaml file containing the desired hyperparameters.",
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
        "--model",
        type=str,
        required=True,
        help="The desired model architecture.",
    )
    parser.add_argument(
        "-e",
        dest="epochs",
        type=int,
        required=True,
        help="The maximum number of training epochs (will be subjected to early stopping).",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    main()
    print("Finished in:")
    print(datetime.now() - start_time)
