import argparse
import optuna
import os
import torch

from lightning import Trainer, Callback
from lightning import seed_everything as lightning_seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from operator import itemgetter
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from sklearn.model_selection import KFold, train_test_split
from torch_geometric import transforms as T
from torch_geometric import seed_everything as geometric_seed_everything
from torch_geometric.loader import DataLoader

from awesom import (
    SOM,
    GNN,
    M1,
    M2,
    M3,
    M4,
    M5,
    M6,
    M7,
    M8,
    M9,
    M10,
    M11,
    M12,
    ValidationMetrics,
)

model_dict = {
    "M1": M1,
    "M2": M2,
    "M3": M3,
    "M4": M4,
    "M5": M5,
    "M6": M6,
    "M7": M7,
    "M8": M8,
    "M9": M9,
    "M10": M10,
    "M11": M11,
    "M12": M12,
}


class PatchedCallback(PyTorchLightningPruningCallback, Callback):
    pass


def run_train():
    # lightning_seed_everything(42)
    # geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    data = SOM(root=args.inputFolder, transform=T.ToUndirected())  # , transform=T.Distance(norm=False)
    train_data, val_data = train_test_split(data, test_size=1/9, random_state=42)
    print(f"Number of training instances: {len(train_data)}")
    print(f"Number of validation instances: {len(val_data)}")

    validation_outputs = {}

    for mid in range(args.ensembleSize):

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

        def objective(trial):
            model_type = model_dict[args.model]
            params, hyperparams = model_type.get_params(data, trial)
            model = GNN(
                params=params,
                hyperparams=hyperparams,
                class_weights=data.get_class_weights(),
                architecture=args.model,
                threshold=0.5,
            )

            tbl = TensorBoardLogger(
                save_dir=os.path.join(
                    os.path.join(args.outputFolder, f"m{mid}"),
                    "logs",
                ),
                name=f"trial{trial._trial_id}",
                default_hp_metric=False,
            )
            callbacks = [
                EarlyStopping(monitor="val/loss", mode="min", min_delta=0, patience=30),
                PatchedCallback(trial=trial, monitor="val/loss"),
                ModelCheckpoint(
                    filename=f"trial{trial._trial_id}", monitor="val/mcc", mode="max"
                ),
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

            return trainer.callback_metrics["val/mcc"].item()

        if not os.path.exists(os.path.join(args.outputFolder, f"m{mid}")):
            os.makedirs(os.path.join(args.outputFolder, f"m{mid}"))

        storage = "sqlite:///" + args.outputFolder + f"/m{mid}" + "/storage.db"
        pruner = optuna.pruners.MedianPruner(n_min_trials=5, n_warmup_steps=100)
        study = optuna.create_study(
            study_name=f"{args.model}_study",
            direction="maximize",
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=args.numberOptunaTrials, gc_after_trial=True)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("   Number of finished trials: ", len(study.trials))
        print("   Number of pruned trials: ", len(pruned_trials))
        print("   Number of complete trials: ", len(complete_trials))
        print(
            f"Best trial is trial {study.best_trial._trial_id} with validation mcc {study.best_trial.value} and hyperparameters:"
        )
        for key, value in study.best_trial.params.items():
            print("   {}: {}".format(key, value))

        path = f"{args.outputFolder}/m{mid}/logs/trial{study.best_trial._trial_id}/version_0/"
        with open(os.path.join(args.outputFolder, "best_model_paths.txt"), "a") as f:
            f.write(path + "\n")

        # Recompute validation metrics with best model

        for file in os.listdir(os.path.join(path, "checkpoints")):
            if file.endswith(".ckpt"):
                checkpoint_path = os.path.join(os.path.join(path, "checkpoints"), file)
        # for file in os.listdir(path):
        #     if file.endswith(".yaml"):
        #         hparams_file = os.path.join(path, file)

        model = GNN.load_from_checkpoint(checkpoint_path)

        trainer = Trainer(accelerator="auto", logger=False)
        validation_outputs[mid] = trainer.predict(
            model=model, dataloaders=DataLoader(val_data, batch_size=len(val_data))
        )

    ValidationMetrics.compute_and_log_validation_metrics(
        validation_outputs, args.outputFolder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training and testing the model.")

    parser.add_argument(
        "-i",
        dest="inputFolder",
        type=str,
        required=True,
        help="The folder where the input data is stored.",
    )
    parser.add_argument(
        "-o",
        dest="outputFolder",
        type=str,
        required=True,
        help="The name of the folder where the model's checkpoints and the validation metrics will be stored.",
    )
    parser.add_argument(
        "-m",
        dest="model",
        type=str,
        required=True,
        help="The model architecture.",
    )
    parser.add_argument(
        "-e",
        dest="epochs",
        type=int,
        required=True,
        help="The maximum number of training epochs.",
    )
    parser.add_argument(
        "-s",
        dest="ensembleSize",
        type=int,
        required=True,
        help="The number of individual models in the deep ensemble.",
    )
    parser.add_argument(
        "-t",
        dest="numberOptunaTrials",
        type=int,
        required=True,
        help="The number of Optuna trials.",
    )

    args = parser.parse_args()
    run_train()
