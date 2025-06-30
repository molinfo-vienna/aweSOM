import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GINEConv, global_add_pool
from torchmetrics import MatthewsCorrCoef
from tqdm import tqdm

from .gpu_utils import get_device


@dataclass
class EnsemblePredictions:
    """Container for ensemble predictions with clear structure."""

    logits: torch.Tensor  # Shape: (num_models, num_atoms)
    y_trues: torch.Tensor  # Shape: (num_atoms,)
    mol_ids: torch.Tensor  # Shape: (num_atoms,)
    atom_ids: torch.Tensor  # Shape: (num_atoms,)
    descriptions: List[str]  # Length: num_atoms

    def shannon_entropy(self, p: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy."""
        return -(p * torch.log2(p + 1e-14) + (1 - p) * torch.log2(1 - p + 1e-14))

    def get_probabilities(self) -> torch.Tensor:
        """Get probabilities."""
        return torch.sigmoid(self.logits)

    def get_uncertainties(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get aleatoric, epistemic, and total uncertainties."""

        # Total uncertainty (entropy of the mean probability)
        u_tot = self.shannon_entropy(torch.mean(self.get_probabilities(), dim=0))

        # Aleatoric uncertainty (mean of the individual entropies)
        u_ale = torch.mean(self.shannon_entropy(self.get_probabilities()), dim=0)

        # Epistemic uncertainty (difference between total and aleatoric uncertainty)
        u_epi = u_tot - u_ale

        return u_ale, u_epi, u_tot

    def to(self, device: torch.device) -> "EnsemblePredictions":
        """Move the predictions to the specified device."""
        return EnsemblePredictions(
            logits=self.logits.to(device),
            y_trues=self.y_trues.to(device),
            mol_ids=self.mol_ids.to(device),
            atom_ids=self.atom_ids.to(device),
            descriptions=self.descriptions,
        )


class GINEWithContextPooling(nn.Module):
    """
    The modified GINConv operator from the "Strategies for Pre-training Graph Neural Networks" paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Pooling for context
    """

    def __init__(
        self, params: Dict[str, int], hyperparams: Dict[str, Union[int, float]]
    ) -> None:
        super(GINEWithContextPooling, self).__init__()

        self.conv = nn.ModuleList()
        self.batch_norm = nn.ModuleList()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for _ in range(int(hyperparams["num_conv_layers"])):
            self.conv.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        nn.LeakyReLU(),
                        nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, mid_channels),
            BatchNorm(mid_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(mid_channels, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
            x = F.leaky_relu(x)

        # Pooling for context
        x_pool = global_add_pool(x, data.batch)
        num_atoms_per_mol = torch.unique(data.batch, sorted=False, return_counts=True)[
            1
        ]
        x_pool_expanded = torch.repeat_interleave(x_pool, num_atoms_per_mol, dim=0)

        # Concatenate final embedding and pooled representation
        x = torch.cat((x, x_pool_expanded), dim=1)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(cls, trial: optuna.trial.Trial) -> Dict[str, Union[int, float]]:
        learning_rate: float = trial.suggest_float(
            "learning_rate", 1e-6, 1e-3, log=True
        )
        weight_decay: float = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        pos_class_weight: float = trial.suggest_float(
            "pos_class_weight", 2, 3, log=False
        )
        num_conv_layers: int = trial.suggest_int("num_conv_layers", 1, 6, log=False)
        size_conv_layers: int = trial.suggest_int(
            "size_conv_layers", low=64, high=1024, log=True
        )
        size_final_mlp_layers: int = trial.suggest_int(
            "size_final_mlp_layers", low=64, high=1024, log=True
        )

        hyperparams = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pos_class_weight": pos_class_weight,
            "num_conv_layers": num_conv_layers,
            "size_conv_layers": size_conv_layers,
            "size_final_mlp_layers": size_final_mlp_layers,
        }

        return hyperparams


class SOMPredictor(nn.Module):
    """Graph Neural Network for site-of-metabolism prediction with integrated training logic."""

    def __init__(
        self, data_params: Dict[str, int], hyperparams: Dict[str, Union[int, float]]
    ) -> None:
        super().__init__()

        self.device = get_device()

        self.model = GINEWithContextPooling(data_params, hyperparams)
        self.model.to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(hyperparams["pos_class_weight"]).to(self.device)
        )
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )
        self.mcc = MatthewsCorrCoef(task="binary").to(self.device)

        # Store for checkpointing
        self.hyperparams = hyperparams
        self.data_params = data_params

    def forward(self, batch: Data) -> torch.Tensor:
        # Move batch to device
        batch = batch.to(self.device)
        return self.model(batch)

    def train_step(self, batch: Data) -> Tuple[float, float]:
        """Single training step."""
        self.train()
        self.optimizer.zero_grad()

        logits = self(batch)
        loss = self.loss_fn(logits, batch.y.float())
        loss.backward()
        self.optimizer.step()

        mcc = self.mcc(torch.sigmoid(logits), batch.y)

        return loss.item(), mcc.item()

    def val_step(self, batch: Data) -> Tuple[float, float]:
        """Single validation step."""
        self.eval()
        with torch.no_grad():
            logits = self(batch)
            loss = self.loss_fn(logits, batch.y.float())
            mcc = self.mcc(torch.sigmoid(logits), batch.y)
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
    def load(cls, path: str) -> "SOMPredictor":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(checkpoint["data_params"], checkpoint["hyperparams"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model

    def fit(
        self,
        train_loader: DataLoader[Data],
        val_loader: Optional[DataLoader[Data]] = None,
        max_epochs: int = 500,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        patience: int = 20,
    ) -> int:
        """Train the model with integrated training loop. Returns the number of epochs trained."""
        writer = SummaryWriter(log_dir) if log_dir else None

        best_val_loss = float("inf")
        patience_counter = 0
        actual_epochs = 0

        for epoch in tqdm(range(max_epochs)):
            actual_epochs = epoch + 1  # Track actual epochs trained

            # Training
            train_losses, train_mccs = [], []
            for batch in train_loader:
                loss, mcc = self.train_step(batch)
                train_losses.append(loss)
                train_mccs.append(mcc)

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_mcc = sum(train_mccs) / len(train_mccs)

            # Validation
            if val_loader:
                val_losses, val_mccs = [], []
                for batch in val_loader:
                    loss, mcc = self.val_step(batch)
                    val_losses.append(loss)
                    val_mccs.append(mcc)

                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_mcc = sum(val_mccs) / len(val_mccs)

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
                        self.save(os.path.join(checkpoint_dir, "best_model.ckpt"))
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                # No validation - model is trained for as many epochs
                # as specified by the config file (max_epochs), which
                # is set to 500 by default and is optimized during
                # hyperparameter search
                if writer:
                    writer.add_scalar("train/loss", avg_train_loss, epoch)
                    writer.add_scalar("train/mcc", avg_train_mcc, epoch)

                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.save(os.path.join(checkpoint_dir, "best_model.ckpt"))

        if writer:
            writer.close()

        return actual_epochs


def predict_ensemble(
    data: DataLoader[Data], model_paths: List[str]
) -> EnsemblePredictions:
    """Run ensemble predictions and return structured results."""

    models = [SOMPredictor.load(path) for path in model_paths]
    all_logits: List[torch.Tensor] = []

    # Predefine outputs for mypy
    y_trues: Optional[torch.Tensor] = None
    mol_ids: Optional[torch.Tensor] = None
    atom_ids: Optional[torch.Tensor] = None
    descriptions: Optional[List[str]] = None

    with torch.no_grad():
        for i, model in enumerate(models):
            logits_list: List[torch.Tensor] = []
            y_trues_list: List[torch.Tensor] = []
            mol_ids_list: List[torch.Tensor] = []
            atom_ids_list: List[torch.Tensor] = []
            descriptions_list: List[str] = []

            for batch in data:
                pred_logits, pred_y, pred_mol, pred_atom, pred_desc = model.predict(
                    batch
                )
                logits_list.append(pred_logits)
                y_trues_list.append(pred_y)
                mol_ids_list.append(pred_mol)
                atom_ids_list.append(pred_atom)
                descriptions_list.extend(pred_desc)

            model_logits = torch.cat(logits_list, dim=0)
            model_y_trues = torch.cat(y_trues_list, dim=0)
            model_mol_ids = torch.cat(mol_ids_list, dim=0)
            model_atom_ids = torch.cat(atom_ids_list, dim=0)

            all_logits.append(model_logits)

            if i == 0:
                y_trues = model_y_trues
                mol_ids = model_mol_ids
                atom_ids = model_atom_ids
                descriptions = descriptions_list

    assert (
        y_trues is not None
        and mol_ids is not None
        and atom_ids is not None
        and descriptions is not None
    )

    ensemble_logits = torch.stack(all_logits, dim=0)

    return EnsemblePredictions(
        logits=ensemble_logits,
        y_trues=y_trues,
        mol_ids=mol_ids,
        atom_ids=atom_ids,
        descriptions=descriptions,
    )
