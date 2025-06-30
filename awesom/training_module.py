import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torchmetrics import MatthewsCorrCoef
from tqdm import tqdm

from awesom.models import M7


class GNN(nn.Module):
    """Graph Neural Network for site-of-metabolism prediction."""

    def __init__(
        self, data_params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super().__init__()

        self.model = M7(data_params, hyperparams)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(hyperparams["pos_class_weight"])
        )
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )
        self.mcc = MatthewsCorrCoef(task="binary")

        # Store for checkpointing
        self.hyperparams: dict[str, Union[int, float]] = hyperparams
        self.data_params: dict[str, int] = data_params

    def forward(self, batch: Data) -> torch.Tensor:
        return self.model(batch)

    def train_step(self, batch: Data) -> Tuple[float, float]:
        """Single training step."""
        self.train()
        self.optimizer.zero_grad()

        logits = self(batch)
        loss = self.loss_fn(logits, batch.y.float())
        loss.backward()
        self.optimizer.step()

        preds = torch.sigmoid(logits)
        mcc = self.mcc(preds, batch.y)

        return loss.item(), mcc.item()

    def val_step(self, batch: Data) -> Tuple[float, float]:
        """Single validation step."""
        self.eval()
        with torch.no_grad():
            logits = self(batch)
            loss = self.loss_fn(logits, batch.y.float())
            preds = torch.sigmoid(logits)
            mcc = self.mcc(preds, batch.y)
            return loss.item(), mcc.item()

    def predict(
        self, batch: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Prediction step."""
        self.eval()
        with torch.no_grad():
            logits = self(batch)
            return logits, batch.y, batch.mol_id, batch.atom_id, batch.description

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hyperparams": self.hyperparams,
                "data_params": self.data_params,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "GNN":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(checkpoint["data_params"], checkpoint["hyperparams"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model


def train_model(
    model: GNN,
    train_loader: DataLoader[Data],
    val_loader: Optional[DataLoader[Data]] = None,
    max_epochs: int = 500,
    log_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    patience: int = 20,
) -> None:
    """Simple training function."""

    # Setup logging
    writer: Optional[SummaryWriter] = SummaryWriter(log_dir) if log_dir else None

    best_val_loss: float = float("inf")
    best_train_loss: float = float("inf")
    patience_counter: int = 0

    for epoch in tqdm(range(max_epochs)):
        # Training
        train_losses: List[float] = []
        train_mccs: List[float] = []
        for batch in train_loader:
            loss, mcc = model.train_step(batch)
            train_losses.append(loss)
            train_mccs.append(mcc)

        avg_train_loss: float = sum(train_losses) / len(train_losses)
        avg_train_mcc: float = sum(train_mccs) / len(train_mccs)

        # Validation
        if val_loader:
            val_losses: List[float] = []
            val_mccs: List[float] = []
            for batch in val_loader:
                loss, mcc = model.val_step(batch)
                val_losses.append(loss)
                val_mccs.append(mcc)

            avg_val_loss: float = sum(val_losses) / len(val_losses)
            avg_val_mcc: float = sum(val_mccs) / len(val_mccs)

            # Logging
            if writer:
                writer.add_scalar("train/loss", avg_train_loss, epoch)
                writer.add_scalar("train/mcc", avg_train_mcc, epoch)
                writer.add_scalar("val/loss", avg_val_loss, epoch)
                writer.add_scalar("val/mcc", avg_val_mcc, epoch)

            # Early stopping and checkpointing based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save(os.path.join(checkpoint_dir, "best_model.ckpt"))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            # No validation - use training loss for checkpointing
            if writer:
                writer.add_scalar("train/loss", avg_train_loss, epoch)
                writer.add_scalar("train/mcc", avg_train_mcc, epoch)

            # Save checkpoint based on training loss
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                patience_counter = 0
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save(os.path.join(checkpoint_dir, "best_model.ckpt"))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if writer:
        writer.close()


def predict_ensemble(
    data: DataLoader[Data], model_paths: List[str]
) -> List[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]]
]:
    """Run ensemble predictions."""
    predictions: List[
        List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
        ]
    ] = []

    for model_path in model_paths:
        model = GNN.load(model_path)
        model_preds: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
        ] = []

        for batch in data:
            pred = model.predict(batch)
            model_preds.append(pred)

        predictions.append(model_preds)

    return predictions
