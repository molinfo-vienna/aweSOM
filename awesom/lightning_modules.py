import torch

from lightning import LightningModule
from torchmetrics import AUROC, MatthewsCorrCoef

from awesom.models import M1, M2, M3, M4, M7, M9, M11, M12

MODELS = {
    "M1": M1,
    "M2": M2,
    "M3": M3,
    "M4": M4,
    "M7": M7,
    "M9": M9,
    "M11": M11,
    "M12": M12,
}


class GNN(LightningModule):
    """
    Parent class for the graph neural network models.

    Args:
        params (dict): dictionary of parameters
        hyperparams (dict): dictionary of hyperparameters
        architecture (str): model architecture
    """

    def __init__(
        self,
        params,
        hyperparams,
        architecture,
    ) -> None:
        super(GNN, self).__init__()

        self.save_hyperparameters()

        self.loss_function = torch.nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=torch.tensor(2.8, dtype=torch.float32)
        )

        self.model = MODELS[architecture](params, hyperparams)

        self.learning_rate = hyperparams["learning_rate"]

        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.train_mcc = MatthewsCorrCoef(task="binary")
        self.val_mcc = MatthewsCorrCoef(task="binary")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    def step(self, batch):
        logits = self.model(batch)
        loss = self.loss_function(logits, batch.y.float())
        return loss, logits

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams,
            {
                "train/loss": 1,
                "train/auroc": 0,
                "train/mcc": 0,
                "val/loss": 1,
                "val/auroc": 0,
                "val/mcc": 0,
            },
        )

    def training_step(self, batch, batch_idx):
        loss, logits = self.step(batch)
        y_hats = torch.sigmoid(logits)
        self.train_auroc(y_hats, batch.y)
        self.train_mcc(y_hats, batch.y)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "train/auroc",
            self.train_auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "train/mcc",
            self.train_mcc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.step(batch)
        y_hats = torch.sigmoid(logits)
        self.val_auroc(y_hats, batch.y)
        self.val_mcc(y_hats, batch.y)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "val/auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "val/mcc",
            self.val_mcc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

    def predict_step(self, batch, batch_idx):
        logits = self.model(batch)
        return logits, batch.y, batch.mol_id, batch.atom_id
