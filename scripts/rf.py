import argparse
import os
import torch
import yaml

from datetime import datetime
from lightning import Trainer, seed_everything
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

from awesom import LabeledData, GNN, RFMetrics


def main():
    seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)

    train_data = LabeledData(root=args.trainDataPath, transform=T.ToUndirected())
    test_data = LabeledData(root=args.testDataPath, transform=T.ToUndirected())

    print(f"Number of training molecules: {len(train_data)}")
    print(f"Number of test molecules: {len(test_data)}")

    checkpoints_path = Path(args.checkpointsPath, "lightning_logs")
    version_paths = [
        Path(checkpoints_path, f"version_{i}")
        for i, _ in enumerate(os.listdir(checkpoints_path))
    ]

    checkpoint_path = [
        Path(Path(version_paths[0], "checkpoints"), file)
        for file in os.listdir(Path(version_paths[0], "checkpoints"))
        if file.endswith(".ckpt")
    ][0]
    hparams = yaml.safe_load(Path(version_paths[0], "hparams.yaml").read_text())

    model = GNN(
        params=hparams["params"],
        hyperparams=hparams["hyperparams"],
        architecture=hparams["architecture"],
        pos_weight=hparams["pos_weight"],
    )

    model = GNN.load_from_checkpoint(checkpoint_path)

    # Disactivate the final (classification) layer
    model.model.final = torch.nn.Identity()

    trainer = Trainer(accelerator="auto", logger=False)

    # Get the embeddings for the training and test data
    predictions_train = trainer.predict(
        model=model, dataloaders=DataLoader(train_data, batch_size=len(train_data))
    )
    predictions_test = trainer.predict(
        model=model, dataloaders=DataLoader(test_data, batch_size=len(test_data))
    )

    embeddings_train = predictions_train[0][0].view(
        predictions_train[0][1].shape[0], -1
    )
    embeddings_test = predictions_test[0][0].view(predictions_test[0][1].shape[0], -1)
    y_train = predictions_train[0][1]
    y_test = predictions_test[0][1]
    mol_ids_train = predictions_train[0][2]
    mol_ids_test = predictions_test[0][2]
    atom_ids_train = predictions_train[0][3]
    atom_ids_test = predictions_test[0][3]

    # Initialize the random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the random forest classifier
    rf.fit(embeddings_train, y_train)
    # Predict the SoMs for the test data with the random forest classifier
    predictions = rf.predict_proba(embeddings_test)
    # Compute and log the test metrics
    RFMetrics.compute_and_log_metrics(
        torch.tensor(predictions[:, 1]),
        y_test,
        mol_ids_test,
        atom_ids_test,
        args.outputPath,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Sandbox script for trying whether replacing the classification module of the GNN with an RF could be benefical."
    )

    parser.add_argument(
        "-itr",
        dest="trainDataPath",
        type=str,
        required=True,
        help="The path to the training data.",
    )
    parser.add_argument(
        "-ite",
        dest="testDataPath",
        type=str,
        required=True,
        help="The path to the test data.",
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

    args = parser.parse_args()

    start_time = datetime.now()
    main()
    print("Finished in:")
    print(datetime.now() - start_time)
