import torch

from lightning import LightningModule
from torchmetrics import AUROC, MatthewsCorrCoef

from awesom.models import M1, M2, M3, M4, M5, M7, M9, M11, M12, M13
from awesom.stochastic_loss import StochasticLoss

MODELS = {
    "M1": M1,
    "M2": M2,
    "M3": M3,
    "M4": M4,
    "M5": M5,
    "M7": M7,
    "M9": M9,
    "M11": M11,
    "M12": M12,
    "M13": M13,
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

        self.loss_function = StochasticLoss(reduction="mean")

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
        logits, stddevs = self.model(batch)
        loss = self.loss_function(logits, stddevs, batch.y.float())
        return loss, logits, stddevs

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
        loss, logits, _ = self.step(batch)
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
        loss, logits, _ = self.step(batch)
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
        logits, stddevs = self.model(batch)
        return logits, stddevs, batch.y, batch.mol_id, batch.atom_id


class EnsembleGNN(LightningModule):
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
        super(EnsembleGNN, self).__init__()

        self.save_hyperparameters()

        self.number_monte_carlo_samples = 10

        self.loss_function = StochasticLoss(reduction="mean")

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
        loss_lst = torch.empty(
            self.number_monte_carlo_samples, dtype=torch.float32, device=self.device
        )
        logits_lst = torch.empty(
            (self.number_monte_carlo_samples, batch.y.size(0)),
            dtype=torch.float32,
            device=self.device,
        )
        stddevs_lst = torch.empty(
            (self.number_monte_carlo_samples, batch.y.size(0)),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(self.number_monte_carlo_samples):
            logits, stddevs = self.model(batch)
            loss = self.loss_function(logits, stddevs, batch.y.float())
            loss_lst[i] = loss
            logits_lst[i, :] = logits
            stddevs_lst[i, :] = stddevs
        return torch.sum(loss), logits_lst, stddevs_lst

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
        loss, logits, _ = self.step(batch)
        y_hats = torch.sigmoid(logits)
        y_hats_avg = torch.mean(y_hats, dim=0)
        self.train_auroc(y_hats_avg, batch.y)
        self.train_mcc(y_hats_avg, batch.y)
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
        loss, logits, _ = self.step(batch)
        y_hats = torch.sigmoid(logits)
        y_hats_avg = torch.mean(y_hats, dim=0)
        self.val_auroc(y_hats_avg, batch.y)
        self.val_mcc(y_hats_avg, batch.y)
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
        logits_lst = torch.empty(
            (self.number_monte_carlo_samples, batch.y.size(0)),
            dtype=torch.float32,
            device=self.device,
        )
        stddevs_lst = torch.empty(
            (self.number_monte_carlo_samples, batch.y.size(0)),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(self.number_monte_carlo_samples):
            logits, stddevs = self.model(batch)
            logits_lst[i, :] = logits
            stddevs_lst[i, :] = stddevs
        return logits_lst, stddevs_lst, batch.y, batch.mol_id, batch.atom_id
