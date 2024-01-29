import argparse
import optuna
import os
import torch

from lightning import Trainer, Callback, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from operator import itemgetter
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from sklearn.model_selection import KFold
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
    ValidationMetrics,
)

NFOLDS = 10
BATCHSIZE = 64

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
}


class PatchedCallback(PyTorchLightningPruningCallback, Callback):
    pass


def run_train():
    seed_everything(42)
    geometric_seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    data = SOM(
        root=args.inputFolder, transform=T.ToUndirected()
    ).shuffle()  # , transform=T.Distance(norm=False)
    print(f"Number of training graphs: {len(data)}")

    validation_outputs = {}

    fold = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    for fold_id, (train_idx, val_idx) in enumerate(fold.split(range(len(data)))):
        train_data = itemgetter(*train_idx)(data)
        val_data = itemgetter(*val_idx)(data)
        train_loader = DataLoader(train_data, batch_size=len(data))
        val_loader = DataLoader(val_data, batch_size=len(data))

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
                    os.path.join(args.outputFolder, f"fold{fold_id}"),
                    "logs",
                ),
                name=f"trial{trial._trial_id}",
                default_hp_metric=False,
            )
            callbacks = [
                EarlyStopping(monitor="val/loss", mode="min", min_delta=0, patience=30),
                PatchedCallback(trial=trial, monitor="val/loss"),
                ModelCheckpoint(
                    filename=f"trial{trial._trial_id}", monitor="val/loss", mode="min"
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

            return trainer.callback_metrics["val/loss"].item()

        if not os.path.exists(os.path.join(args.outputFolder, f"fold{fold_id}")):
            os.makedirs(os.path.join(args.outputFolder, f"fold{fold_id}"))

        storage = "sqlite:///" + args.outputFolder + f"/fold{fold_id}" + "/storage.db"
        pruner = optuna.pruners.MedianPruner(n_min_trials=5, n_warmup_steps=5000)
        study = optuna.create_study(
            study_name=f"{args.model}_study",
            direction="minimize",
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=args.numberTrials, gc_after_trial=True)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("   Number of finished trials: ", len(study.trials))
        print("   Number of pruned trials: ", len(pruned_trials))
        print("   Number of complete trials: ", len(complete_trials))
        print(
            f"Best trial is trial {study.best_trial._trial_id} with validation loss {study.best_trial.value} and hyperparameters:"
        )
        for key, value in study.best_trial.params.items():
            print("   {}: {}".format(key, value))

        path = f"{args.outputFolder}/fold{fold_id}/logs/trial{study.best_trial._trial_id}/version_0/"
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
        validation_outputs[fold_id] = trainer.predict(
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
        help="The desired model architecture.",
    )
    parser.add_argument(
        "-e",
        dest="epochs",
        type=int,
        required=True,
        help="The maximum number of training epochs.",
    )
    parser.add_argument(
        "-nt",
        dest="numberTrials",
        type=int,
        required=True,
        help="The number of Optuna trials.",
    )

    args = parser.parse_args()
    run_train()
