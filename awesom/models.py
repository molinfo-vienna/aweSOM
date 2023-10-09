import torch
from torch_geometric.nn import (
    GATv2Conv,
    GINConv,
    GINEConv,
    NNConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from typing import List, Optional, Tuple, Union
from lightning import LightningModule
from torchmetrics import MatthewsCorrCoef, AUROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torch_geometric.nn.norm import LayerNorm


__all__ = ["GNN", "GATv2", "GIN", "GINE", "NN"]


class GNN(LightningModule):
    def __init__(
        self,
        params,
        hyperparams,
    ) -> None:
        super(GNN, self).__init__()

        self.threshold = 0.2

        self.loss_function = torch.nn.BCELoss()

        self.train_mcc = MatthewsCorrCoef(task="binary", threshold=self.threshold)
        self.train_auroc = AUROC(task="binary")

        self.val_mcc = MatthewsCorrCoef(task="binary", threshold=self.threshold)
        self.val_auroc = AUROC(task="binary")

        self.test_mcc = MatthewsCorrCoef(task="binary", threshold=self.threshold)
        self.test_auroc = AUROC(task="binary")
        self.test_precision = BinaryPrecision(threshold=self.threshold)
        self.test_recall = BinaryRecall(threshold=self.threshold)

        self.learning_rate = hyperparams["learning_rate"]
        self.weight_decay = hyperparams["weight_decay"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "train/loss": 0,
                "train/mcc": 0,
                "train/auroc": 0,
                "val/loss": 0,
                "val/mcc": 0,
                "val/auroc": 0,
            },
        )

    def step(self, batch):
        y_hat = self(batch, batch.batch).flatten().to(float)
        loss = self.loss_function(y_hat, batch.y.to(float))
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch)
        self.train_mcc(y_hat, batch.y)
        self.train_auroc(y_hat, batch.y)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        self.log(
            "train/mcc",
            self.train_mcc,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=len(batch),
        )
        self.log(
            "train/auroc",
            self.train_auroc,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=len(batch),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch)
        self.val_mcc(y_hat, batch.y)
        self.val_auroc(y_hat, batch.y)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
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
        self.log(
            "val/auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

    def test_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch)
        self.test_mcc(y_hat, batch.y)
        self.test_auroc(y_hat, batch.y)
        self.test_precision(y_hat, batch.y)
        self.test_recall(y_hat, batch.y)
        self.log(
            "test/mcc",
            self.test_mcc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "test/auroc",
            self.test_auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "test/precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )
        self.log(
            "test/recall",
            self.test_recall,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

    def predict_step(self, batch, batch_idx):
        _, y_hat = self.step(batch)
        return y_hat, batch.y, batch.mol_id, batch.atom_id


class GATv2(GNN):
    """The GATv2 operator from the “How Attentive are Graph Attention Networks?” paper,
    which fixes the static attention problem of the standard GATConv layer.
    Since the linear layers in the standard GAT are applied right after each other,
    the ranking of attended nodes is unconditioned on the query node.
    In contrast, in GATv2, every node can attend to any other node.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv
    """

    def __init__(
        self,
        params,
        hyperparams,
    ) -> None:
        super(GATv2, self).__init__(params, hyperparams)

        self.save_hyperparameters()

        self.conv = torch.nn.ModuleList()
        in_channels = params["num_node_features"]
        for out_channels in hyperparams["size_conv_layers"]:
            self.conv.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=hyperparams["heads"],
                    negative_slope=hyperparams["negative_slope"],
                    dropout=hyperparams["dropout"],
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels * hyperparams["heads"]

        self.classifier = torch.nn.ModuleList()
        in_channels = (
            sum(hyperparams["size_conv_layers"]) + hyperparams["size_conv_layers"][-1]
        ) * hyperparams["heads"]
        for out_channels in hyperparams["size_classify_layers"]:
            self.classifier.append(torch.nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        self.final = torch.nn.Linear(in_channels, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

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
            h = self.leaky_relu(h)
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
        h = self.final(h)

        return torch.sigmoid(h)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        n_conv_layers = trial.suggest_int("n_conv_layers", 1, 5)
        size_conv_layers = [
            trial.suggest_int(f"size_conv_layers_{i}", 16, 512)
            for i in range(n_conv_layers)
        ]
        n_classify_layers = trial.suggest_int("n_classify_layers", 1, 5)
        size_classify_layers = [
            trial.suggest_int(f"size_classify_layers_{i}", 16, 512)
            for i in range(n_classify_layers)
        ]
        heads = trial.suggest_int("heads", 2, 8)
        negative_slope = trial.suggest_float("negative_slope", 0.1, 0.9)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_conv_layers=n_conv_layers,
            size_conv_layers=size_conv_layers,
            n_classify_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
        )

        return params, hyperparams


class GIN(GNN):
    """The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv
    """

    def __init__(
        self,
        params,
        hyperparams,
    ) -> None:
        super(GIN, self).__init__(params, hyperparams)

        self.save_hyperparameters()

        self.conv = torch.nn.ModuleList()
        in_channels = params["num_node_features"]
        for out_channels in hyperparams["size_conv_layers"]:
            self.conv.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            in_channels, hyperparams["size_hidden_network"]
                        ),
                        torch.nn.BatchNorm1d(
                            hyperparams["size_hidden_network"],
                            hyperparams["size_hidden_network"],
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            hyperparams["size_hidden_network"], out_channels
                        ),
                        torch.nn.LeakyReLU(),
                    ),
                    train_eps=True,
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

        self.final = torch.nn.Linear(in_channels, 1)

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
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
            h = self.dropout(h)
        h = self.final(h)

        return torch.sigmoid(h)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        n_conv_layers = trial.suggest_int("n_conv_layers", 1, 5)
        size_conv_layers = [
            trial.suggest_int(f"size_conv_layers_{i}", 16, 512)
            for i in range(n_conv_layers)
        ]
        n_classify_layers = trial.suggest_int("n_classify_layers", 1, 5)
        size_classify_layers = [
            trial.suggest_int(f"size_classify_layers_{i}", 16, 512)
            for i in range(n_classify_layers)
        ]
        size_hidden_network = trial.suggest_int("size_hidden_network", 16, 512)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_conv_layers=n_conv_layers,
            size_conv_layers=size_conv_layers,
            n_classify_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
            size_hidden_network=size_hidden_network,
            dropout=dropout,
        )

        return params, hyperparams


class GINE(GNN):
    """The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
    """

    def __init__(self, params, hyperparams) -> None:
        super(GINE, self).__init__(params, hyperparams)

        self.save_hyperparameters()

        self.conv = torch.nn.ModuleList()
        in_channels = params["num_node_features"]
        for out_channels in hyperparams["size_conv_layers"]:
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            in_channels, hyperparams["size_hidden_network"]
                        ),
                        LayerNorm(
                            hyperparams["size_hidden_network"],
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            hyperparams["size_hidden_network"], out_channels
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

        self.final = torch.nn.Linear(in_channels, 1)

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

        return torch.sigmoid(h)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        n_conv_layers = trial.suggest_int("n_conv_layers", 1, 5)
        size_conv_layers = [
            trial.suggest_int(f"size_conv_layers_{i}", 16, 512)
            for i in range(n_conv_layers)
        ]
        n_classify_layers = trial.suggest_int("n_classify_layers", 1, 5)
        size_classify_layers = [
            trial.suggest_int(f"size_classify_layers_{i}", 16, 512)
            for i in range(n_classify_layers)
        ]
        size_hidden_network = trial.suggest_int("size_hidden_network", 16, 512)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_conv_layers=n_conv_layers,
            size_conv_layers=size_conv_layers,
            n_classify_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
            size_hidden_network=size_hidden_network,
            dropout=dropout,
        )

        return params, hyperparams


class NN(GNN):
    """The continuous kernel-based convolutional operator from the “Neural Message Passing for Quantum Chemistry” paper
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.NNConv.html#torch_geometric.nn.conv.NNConv
    """

    def __init__(
        self,
        params,
        hyperparams,
    ) -> None:
        super(NN, self).__init__(params, hyperparams)

        self.save_hyperparameters()

        self.conv = torch.nn.ModuleList()
        in_channels = params["num_node_features"]
        for out_channels in hyperparams["size_conv_layers"]:
            self.conv.append(
                NNConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    edge_dim=params["num_edge_features"],
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(
                            params["num_edge_features"],
                            hyperparams["size_hidden_network"],
                        ),
                        torch.nn.BatchNorm1d(
                            hyperparams["size_hidden_network"],
                            hyperparams["size_hidden_network"],
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            hyperparams["size_hidden_network"],
                            in_channels * out_channels,
                        ),
                        torch.nn.LeakyReLU(),
                    ),
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

        self.final = torch.nn.Linear(in_channels, 1)

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

        return torch.sigmoid(h)

    @classmethod
    def get_params(self, data, trial):
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        n_conv_layers = trial.suggest_int("n_conv_layers", 1, 5)
        size_conv_layers = [
            trial.suggest_int(f"size_conv_layers_{i}", 16, 512)
            for i in range(n_conv_layers)
        ]
        n_classify_layers = trial.suggest_int("n_classify_layers", 1, 5)
        size_classify_layers = [
            trial.suggest_int(f"size_classify_layers_{i}", 16, 512)
            for i in range(n_classify_layers)
        ]
        size_hidden_network = trial.suggest_int("size_hidden_network", 16, 512)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)

        params = dict(
            num_node_features=data.num_node_features,
            num_edge_features=data.num_edge_features,
        )

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_conv_layers=n_conv_layers,
            size_conv_layers=size_conv_layers,
            n_classify_layers=n_classify_layers,
            size_classify_layers=size_classify_layers,
            size_hidden_network=size_hidden_network,
            dropout=dropout,
        )

        return params, hyperparams
