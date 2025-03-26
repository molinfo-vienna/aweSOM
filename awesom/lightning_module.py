from typing import Tuple, Union

import torch
from lightning import LightningModule
from torch_geometric.data import Batch
from torchmetrics import MatthewsCorrCoef

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
        params: dict[str, int],
        hyperparams: dict[str, Union[int, float]],
        architecture: str,
    ) -> None:
        super(GNN, self).__init__()

        self.save_hyperparameters()

        self.model = MODELS[architecture](params, hyperparams)
        self.pos_class_weight: float = hyperparams["pos_class_weight"]
        self.learning_rate: float = hyperparams["learning_rate"]
        self.weight_decay: float = hyperparams["weight_decay"]

        self.loss_function = torch.nn.BCEWithLogitsLoss(
            reduction="mean",
            pos_weight=torch.tensor(self.pos_class_weight, dtype=torch.float32),
        )

        self.train_mcc = MatthewsCorrCoef(task="binary")
        self.val_mcc = MatthewsCorrCoef(task="binary")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def step(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(batch)
        loss = self.loss_function(logits, batch.y.float())
        return loss, logits

    def on_train_start(self) -> None:
        if self.logger is not None:
            if isinstance(self.hparams, dict):
                hyperparams = self.hparams
            else:
                hyperparams = (
                    vars(self.hparams) if hasattr(self.hparams, "__dict__") else {}
                )
            self.logger.log_hyperparams(
                hyperparams,
                {
                    "train/loss": 1,
                    "train/mcc": 0,
                    "val/loss": 1,
                    "val/mcc": 0,
                },
            )

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, logits = self.step(batch)
        y_hats = torch.sigmoid(logits)
        self.train_mcc(y_hats, batch.y)
        self.log(
            "train/loss",
            loss,
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

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        loss, logits = self.step(batch)
        y_hats = torch.sigmoid(logits)
        self.val_mcc(y_hats, batch.y)
        self.log(
            "val/loss",
            loss,
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

    def predict_step(
        self, batch: Batch, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        logits = self.model(batch)
        return logits, batch.y, batch.mol_id, batch.atom_id, batch.description
