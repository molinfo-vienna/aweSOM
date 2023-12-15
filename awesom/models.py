import torch
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GINConv,
    GINEConv,
    MFConv,
    ChebConv,
    global_add_pool,
    global_max_pool,
)
from typing import List, Optional, Tuple, Union
from lightning import LightningModule
from torchmetrics import AUROC, MatthewsCorrCoef
from torch_geometric.nn.norm import LayerNorm
from torch_geometric import transforms as T


__all__ = ["GNN", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "GINE", "GINED", "MF", "Cheb"]


class GNN(LightningModule):
    def __init__(
        self,
        params,
        hyperparams,
        class_weights,
        architecture,
    ) -> None:
        super(GNN, self).__init__()

        self.save_hyperparameters()
        self.model = architecture(params, hyperparams, class_weights)

        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights, reduction="mean"
        )

        self.train_auroc = AUROC(task="multiclass", num_classes=2)
        self.val_auroc = AUROC(task="multiclass", num_classes=2)
        self.train_mcc = MatthewsCorrCoef(
            task="multiclass", num_classes=2, threshold=0.5
        )
        self.val_mcc = MatthewsCorrCoef(task="multiclass", num_classes=2, threshold=0.5)

        self.learning_rate = hyperparams["learning_rate"]
        self.weight_decay = hyperparams["weight_decay"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
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
        y_hat = self.model(batch, batch.batch)
        loss = self.loss_function(y_hat, batch.y)
        return loss, y_hat

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {"train/loss": 1, 
                                                   "train/auroc": 0,
                                                   "train/mcc": 0,
                                                   "val/loss": 1, 
                                                   "val/auroc": 0,
                                                   "val/mcc": 0})

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch)
        self.train_auroc(y_hat, batch.y)
        self.train_mcc(y_hat, batch.y)
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
        loss, y_hat = self.step(batch)
        self.val_auroc(y_hat, batch.y)
        self.val_mcc(y_hat, batch.y)
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
        _, y_hat = self.step(batch)
        return y_hat, batch.y, batch.mol_id, batch.atom_id


class M1(torch.nn.Module):

    def __init__(self, params, hyperparams, class_weights) -> None:
        super(M1, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        self.activation,
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            self.batch_norm.append(BatchNorm(out_channels))
            in_channels = out_channels

        self.final = torch.nn.Linear(out_channels, class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:

        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i < len(self.conv)-1:
                x = batch_norm(x)
                x = self.activation(x)

        x = self.final(x)

        return torch.softmax(x, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int(f"size_conv_layers", 32, 512, log=True)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
        )

        return params, hyperparams


class M2(torch.nn.Module):

    def __init__(self, params, hyperparams, class_weights) -> None:
        super(M2, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        self.activation,
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                )
            )
            self.batch_norm.append(BatchNorm(out_channels))
            in_channels = out_channels

        self.final = torch.nn.Linear(out_channels, class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:

        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index)
            if i < len(self.conv)-1:
                x = batch_norm(x)
                x = self.activation(x)

        x = self.final(x)

        return torch.softmax(x, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int(f"size_conv_layers", 32, 512, log=True)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
        )

        return params, hyperparams


class M3(torch.nn.Module):

    def __init__(self, params, hyperparams, class_weights) -> None:
        super(M3, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        self.activation,
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            self.batch_norm.append(BatchNorm(out_channels))
            in_channels = out_channels

        self.final = torch.nn.Linear(out_channels * hyperparams["num_conv_layers"], class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:

        # Compute node intermediate embeddings
        h = []
        x = data.x
        for conv, batch_norm in zip(self.conv, self.batch_norm):
            x = conv(x, data.edge_index, data.edge_attr)
            x = batch_norm(x)
            x = self.activation(x)
            h.append(x)
            
        # Concatenate embeddings (jumping knowledge)
        h = torch.cat(h, dim=1)

        # Classify
        h = self.final(h)

        return torch.softmax(h, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int(f"size_conv_layers", 32, 512, log=True)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
        )

        return params, hyperparams
    

class M4(torch.nn.Module):

    def __init__(self, params, hyperparams, class_weights) -> None:
        super(M4, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        self.activation,
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            self.batch_norm.append(BatchNorm(out_channels))
            in_channels += out_channels

        self.final = torch.nn.Linear(out_channels * hyperparams["num_conv_layers"] + params["num_node_features"], class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:

        # Compute node intermediate embeddings
        x = data.x
        for conv, batch_norm in zip(self.conv, self.batch_norm):
            h = conv(x, data.edge_index, data.edge_attr)
            h = batch_norm(h)
            h = self.activation(h)
            x = torch.cat((x,h), dim=1)

        # Classify
        x = self.final(x)

        return torch.softmax(x, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int(f"size_conv_layers", 32, 512, log=True)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
        )

        return params, hyperparams


class M5(torch.nn.Module):

    def __init__(self, params, hyperparams, class_weights) -> None:
        super(M5, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        self.activation,
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            self.batch_norm.append(BatchNorm(out_channels))
            in_channels = out_channels

        self.final = torch.nn.Linear(out_channels * 2, class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:

        # Convolutions
        x = data.x
        for conv, batch_norm in zip(self.conv, self.batch_norm):
            x = conv(x, data.edge_index, data.edge_attr)
            x = batch_norm(x)
            x = self.activation(x)

        # Pooling for context
        x_pool = global_add_pool(x, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        x_pool_expanded = torch.repeat_interleave(x_pool, num_atoms_per_mol, dim=0)

        # Concatenate final embedding and pooled representation
        x = torch.cat((x, x_pool_expanded), dim=1)

        # Classification
        x = self.final(x)

        return torch.softmax(x, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int(f"size_conv_layers", 32, 512, log=True)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
        )

        return params, hyperparams


class M6(torch.nn.Module):

    def __init__(self, params, hyperparams, class_weights) -> None:
        super(M6, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        self.activation,
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            self.batch_norm.append(BatchNorm(out_channels))
            in_channels = out_channels

        self.final = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, class_weights.shape[0]),
                BatchNorm(class_weights.shape[0]),
                self.activation,
                torch.nn.Linear(class_weights.shape[0], class_weights.shape[0]),
                ),
            train_eps=True,
            edge_dim=params["num_edge_features"],
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:

        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i < len(self.conv)-1:
                x = batch_norm(x)
                x = self.activation(x)

        x = self.final(x, data.edge_index, data.edge_attr)

        return torch.softmax(x, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int(f"size_conv_layers", 32, 512, log=True)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
        )

        return params, hyperparams


class M7(torch.nn.Module):
    """The GATv2 operator from the “How Attentive are Graph Attention Networks?” paper,
    which fixes the static attention problem of the standard GATConv layer.
    Since the linear layers in the standard GAT are applied right after each other,
    the ranking of attended nodes is unconditioned on the query node.
    In contrast, in GATv2, every node can attend to any other node.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv
    """
    def __init__(self, params, hyperparams, class_weights) -> None:
        super(M7, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=hyperparams["heads"],
                    negative_slope=hyperparams["negative_slope"],
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels * hyperparams["heads"]
            self.batch_norm.append(BatchNorm(in_channels))
            
        self.final = torch.nn.Linear(in_channels, class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:

        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i < len(self.conv)-1:
                x = batch_norm(x)
                x = self.activation(x)

        x = self.final(x)

        return torch.softmax(x, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int(f"size_conv_layers", 32, 512, log=True)
        heads = trial.suggest_int("heads", 2, 8)
        negative_slope = trial.suggest_float("negative_slope", 0.1, 0.9)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            heads=heads,
            negative_slope=negative_slope,

        )

        return params, hyperparams


class GINE(torch.nn.Module):
    """The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
    """

    def __init__(self, params, hyperparams, class_weights) -> None:
        super(GINE, self).__init__()

        self.conv = torch.nn.ModuleList()
        in_channels = params["num_node_features"]
        for out_channels in hyperparams["size_conv_layers"]:
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        LayerNorm(
                            out_channels,
                        ),
                        torch.nn.LeakyReLU(),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels

        self.classifier = torch.nn.ModuleList()
        in_channels = (
            sum(hyperparams["size_conv_layers"]) + hyperparams["size_conv_layers"][-1]
        )
        for out_channels in hyperparams["size_classify_layers"]:
            self.classifier.append(torch.nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        self.final = torch.nn.Linear(in_channels, class_weights.shape[0])

        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=hyperparams["dropout"])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        x = data.x
        for layer in self.conv:
            h = layer(x, data.edge_index, data.edge_attr)
            x = h
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
            h = self.dropout(h)
        h = self.final(h)

        return torch.softmax(h, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = [
            trial.suggest_int(f"size_conv_layers_{i}", 64, 512)
            for i in range(num_conv_layers)
        ]
        n_classify_layers = trial.suggest_int("n_classify_layers", 1, 5)
        size_classify_layers = [
            trial.suggest_int(f"size_classify_layers_{i}", 64, 512)
            for i in range(n_classify_layers)
        ]
        dropout = trial.suggest_float("dropout", 0.1, 0.3)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            n_classify_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
            dropout=dropout,
        )

        return params, hyperparams


class GINED(torch.nn.Module):
    """The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
    """

    def __init__(self, params, hyperparams, class_weights) -> None:
        super(GINED, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        self.activation,
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            self.batch_norm.append(BatchNorm(out_channels))
            in_channels = out_channels

        self.final = torch.nn.Linear(out_channels * hyperparams["num_conv_layers"] * 2, class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:

        # Compute node intermediate embeddings
        h = []
        x = data.x
        for conv, batch_norm in zip(self.conv, self.batch_norm):
            x = conv(x, data.edge_index, data.edge_attr)
            x = batch_norm(x)
            x = self.activation(x)
            h.append(x)

        # Concatenate embeddings (jumping knowledge)
        h = torch.cat(h, dim=1)

        # Pooling for context
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_expanded = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings and pooled representation
        h = torch.cat((h, h_pool_expanded), dim=1)

        # Classify
        h = self.final(h)

        return torch.softmax(h, dim=1)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int(f"size_conv_layers", 32, 512, log=True)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
        )

        return params, hyperparams


class MF(torch.nn.Module):
    """The graph neural network operator from the “Convolutional Networks on Graphs for Learning Molecular Fingerprints” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv
    """

    def __init__(
        self,
        params,
        hyperparams,
        class_weights,
    ) -> None:
        super(MF, self).__init__()

        self.conv = torch.nn.ModuleList()
        in_channels = params["num_node_features"]
        for out_channels in hyperparams["size_conv_layers"]:
            self.conv.append(
                MFConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    max_degree=hyperparams["max_degree"],
                )
            )
            in_channels = out_channels

        in_channels = (
            sum(hyperparams["size_conv_layers"]) + hyperparams["size_conv_layers"][-1]
        )
        self.final = torch.nn.Linear(in_channels, class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        x = data.x
        for layer in self.conv:
            h = layer(x, data.edge_index)
            x = h
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.final(h)

        return torch.softmax(h, dim=1)
    
    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = [
            trial.suggest_int(f"size_conv_layers_{i}", 64, 512)
            for i in range(num_conv_layers)
        ]
        max_degree = trial.suggest_int("max_degree", 1, 5)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            max_degree=max_degree,
        )

        return params, hyperparams


class Cheb(torch.nn.Module):
    """The graph neural network operator from the “Convolutional Networks on Graphs for Learning Molecular Fingerprints” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv
    """

    def __init__(
        self,
        params,
        hyperparams,
        class_weights,
    ) -> None:
        super(Cheb, self).__init__()

        self.conv = torch.nn.ModuleList()
        in_channels = params["num_node_features"]
        for out_channels in hyperparams["size_conv_layers"]:
            self.conv.append(
                ChebConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    K=hyperparams["filter_size"],
                )
            )
            in_channels = out_channels

        in_channels = (
            sum(hyperparams["size_conv_layers"]) + hyperparams["size_conv_layers"][-1]
        )
        self.final = torch.nn.Linear(in_channels, class_weights.shape[0])

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        x = data.x
        for layer in self.conv:
            h = layer(x, data.edge_index)
            x = h
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        h = self.final(h)

        return torch.softmax(h, dim=1)
    
    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = [
            trial.suggest_int(f"size_conv_layers_{i}", 64, 512)
            for i in range(num_conv_layers)
        ]
        filter_size = trial.suggest_int("filter_size", 1, 10)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            filter_size=filter_size,
        )

        return params, hyperparams
