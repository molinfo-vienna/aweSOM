from typing import Union

import optuna
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GINConv,
    GINEConv,
    MFConv,
    global_add_pool,
)


class M1(torch.nn.Module):
    """
    The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper.
    """

    def __init__(
        self, params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super(M1, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for _ in range(int(hyperparams["num_conv_layers"])):
            self.conv.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        torch.nn.BatchNorm1d(out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                )
            )
            in_channels = out_channels
            self.batch_norm.append(torch.nn.BatchNorm1d(in_channels))

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            torch.nn.BatchNorm1d(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, mid_channels),
            torch.nn.BatchNorm1d(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
            x = F.leaky_relu(x)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(cls, trial: optuna.trial.Trial) -> dict[str, Union[int, float]]:
        learning_rate: float = trial.suggest_float(
            "learning_rate", 1e-6, 1e-3, log=True
        )
        weight_decay: float = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        pos_class_weight: float = trial.suggest_float(
            "pos_class_weight", 2, 3, log=False
        )
        num_conv_layers: int = int(
            trial.suggest_int("num_conv_layers", 1, 6, log=False)
        )
        size_conv_layers: int = int(
            trial.suggest_int("size_conv_layers", low=64, high=1024, log=True)
        )
        size_final_mlp_layers: int = int(
            trial.suggest_int("size_final_mlp_layers", low=64, high=1024, log=True)
        )

        return {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pos_class_weight": pos_class_weight,
            "num_conv_layers": num_conv_layers,
            "size_conv_layers": size_conv_layers,
            "size_final_mlp_layers": size_final_mlp_layers,
        }


class M2(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    """

    def __init__(
        self, params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super(M2, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for _ in range(int(hyperparams["num_conv_layers"])):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
            x = F.leaky_relu(x)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial: optuna.trial.Trial) -> dict[str, Union[int, float]]:
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

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_class_weight=pos_class_weight,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M3(torch.nn.Module):
    """The GATv2 operator from the “How Attentive are Graph Attention Networks?” paper,
    which fixes the static attention problem of the standard GATConv layer.
    Since the linear layers in the standard GAT are applied right after each other,
    the ranking of attended nodes is unconditioned on the query node.
    In contrast, in GATv2, every node can attend to any other node.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv
    """

    def __init__(
        self, params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super(M3, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for _ in range(int(hyperparams["num_conv_layers"])):
            self.conv.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=hyperparams["heads"],
                    negative_slope=hyperparams["negative_slope"],
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels * int(hyperparams["heads"])
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
            x = F.leaky_relu(x)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial: optuna.trial.Trial) -> dict[str, Union[int, float]]:
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
        heads = trial.suggest_int("heads", 1, 6)
        negative_slope = trial.suggest_float("negative_slope", 0.1, 0.9)

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_class_weight=pos_class_weight,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
            heads=heads,
            negative_slope=negative_slope,
        )

        return hyperparams


class M4(torch.nn.Module):
    """The graph neural network operator from the
    “Convolutional Networks on Graphs for Learning Molecular Fingerprints” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv
    """

    def __init__(
        self, params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super(M4, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for _ in range(int(hyperparams["num_conv_layers"])):
            self.conv.append(
                MFConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    max_degree=hyperparams["max_degree"],
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
            x = F.leaky_relu(x)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial: optuna.trial.Trial) -> dict[str, Union[int, float]]:
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
        max_degree = trial.suggest_int("max_degree", 1, 6)

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_class_weight=pos_class_weight,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
            max_degree=max_degree,
        )

        return hyperparams


class M7(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Pooling for context
    """

    def __init__(
        self, params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super(M7, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for _ in range(int(hyperparams["num_conv_layers"])):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
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
    def get_params(self, trial: optuna.trial.Trial) -> dict[str, Union[int, float]]:
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

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_class_weight=pos_class_weight,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M9(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Molecular features as additional input
    """

    def __init__(
        self, params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super(M9, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])
        for _ in range(int(hyperparams["num_conv_layers"])):
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        self.norm_mol_x = torch.nn.LayerNorm(params["num_mol_features"])

        in_channels = in_channels + params["num_mol_features"]
        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        # Compute atom embeddings
        x_atom = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x_atom = conv(x_atom, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x_atom = batch_norm(x_atom)
            x_atom = F.leaky_relu(x_atom)

        # Normalize molecular features
        x_mol = self.norm_mol_x(data.mol_x)

        # Concatenate atom embeddings and molecular embeddings
        x = torch.cat((x_atom, x_mol), dim=1)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial: optuna.trial.Trial) -> dict[str, Union[int, float]]:
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

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_class_weight=pos_class_weight,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M11(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Skip connections in the style of DenseNet
    """

    def __init__(
        self, params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super(M11, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.activation = torch.nn.LeakyReLU()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for i in range(int(hyperparams["num_conv_layers"])):
            self.batch_norm.append(BatchNorm(in_channels))
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
            in_channels = (i + 1) * out_channels + params["num_node_features"]

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        # Convolutions with skip connections
        h = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            h_norm = batch_norm(h)
            if i == 0:
                h_prime = conv(h_norm, data.edge_index, data.edge_attr)
            else:
                h_act = self.activation(h_norm)
                h_prime = conv(h_act, data.edge_index, data.edge_attr)
            h = torch.cat((h, h_prime), dim=1)

        # Classification
        x = self.classifier(h)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial: optuna.trial.Trial) -> dict[str, Union[int, float]]:
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

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_class_weight=pos_class_weight,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M12(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Skip connections in the style of ResNet
    """

    def __init__(
        self, params: dict[str, int], hyperparams: dict[str, Union[int, float]]
    ) -> None:
        super(M12, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.activation = torch.nn.LeakyReLU()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for _ in range(int(hyperparams["num_conv_layers"])):
            self.batch_norm.append(BatchNorm(in_channels))
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
            in_channels = out_channels

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:
        # Convolutions with skip connections
        h = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            h_norm = batch_norm(h)
            if i == 0:
                h = conv(h_norm, data.edge_index, data.edge_attr)
            else:
                h_act = self.activation(h_norm)
                h_prime = conv(h_act, data.edge_index, data.edge_attr)
                h = torch.add(h, h_prime)

        # Classification
        x = self.classifier(h)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial: optuna.trial.Trial) -> dict[str, Union[int, float]]:
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

        hyperparams = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pos_class_weight=pos_class_weight,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams
