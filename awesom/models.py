import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GINConv,
    GINEConv,
    MFConv,
    ChebConv,
    global_add_pool,
)
from typing import Tuple, Union


class M1(torch.nn.Module):
    """
    The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper.
    https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.conv.GINConv.html
    """

    def __init__(self, params, hyperparams) -> None:
        super(M1, self).__init__()

        self.mode = hyperparams["mode"]

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
                x = F.leaky_relu(x)
            if self.mode == "mcdropout":
                x = F.dropout(x, p=0.3, training=True)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M2(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    """

    def __init__(self, params, hyperparams) -> None:
        super(M2, self).__init__()

        self.mode = hyperparams["mode"]

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
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

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
                x = F.leaky_relu(x)
            if self.mode == "mcdropout":
                x = F.dropout(x, p=0.3, training=True)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
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

    def __init__(self, params, hyperparams) -> None:
        super(M3, self).__init__()

        self.mode = hyperparams["mode"]

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

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

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
                x = F.leaky_relu(x)
            if self.mode == "mcdropout":
                x = F.dropout(x, p=0.3, training=True)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)
        heads = trial.suggest_int("heads", 1, 4)
        negative_slope = trial.suggest_float("negative_slope", 0.1, 0.9)

        hyperparams = dict(
            learning_rate=learning_rate,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
            heads=heads,
            negative_slope=negative_slope,
        )

        return hyperparams


class M4(torch.nn.Module):
    """The graph neural network operator from the “Convolutional Networks on Graphs for Learning Molecular Fingerprints” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv
    """

    def __init__(self, params, hyperparams) -> None:
        super(M4, self).__init__()

        self.mode = hyperparams["mode"]

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                MFConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    max_degree=hyperparams["max_degree"],
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
                x = F.leaky_relu(x)
            if self.mode == "mcdropout":
                x = F.dropout(x, p=0.3, training=True)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        max_degree = trial.suggest_int("max_degree", 1, 6)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            max_degree=max_degree,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M5(torch.nn.Module):
    """The chebyshev spectral graph convolutional operator from the
    “Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ChebConv.html
    """

    def __init__(self, params, hyperparams) -> None:
        super(M5, self).__init__()

        self.mode = hyperparams["mode"]

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
            self.conv.append(
                ChebConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    K=hyperparams["filter_size"],
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
                x = F.leaky_relu(x)
            if self.mode == "mcdropout":
                x = F.dropout(x, p=0.3, training=True)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        filter_size = trial.suggest_int("filter_size", 1, 10)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            filter_size=filter_size,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M7(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Pooling for context
    """

    def __init__(self, params, hyperparams) -> None:
        super(M7, self).__init__()
        
        self.mode = hyperparams["mode"]

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
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

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
                x = F.leaky_relu(x)
            if self.mode == "mcdropout":
                x = F.dropout(x, p=0.3, training=True)

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
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M9(torch.nn.Module):

    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Normalized molecular features as additional input
    """

    def __init__(self, params, hyperparams) -> None:
        super(M9, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]
        for _ in range(hyperparams["num_conv_layers"]):
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
        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Compute atom embeddings
        x_atom = data.x
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x_atom = conv(x_atom, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x_atom = batch_norm(x_atom)
                x_atom = self.activation(x_atom)
            if self.mode == "mcdropout":
                x_atom = F.dropout(x_atom, p=0.3, training=True)

        # Normalize molecular features
        x_mol = self.norm_mol_x(data.mol_x)

        # Concatenate atom embeddings and molecular embeddings
        x = torch.cat((x_atom, x_mol), dim=1)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
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

    def __init__(self, params, hyperparams) -> None:
        super(M11, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for i in range(hyperparams["num_conv_layers"]):
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

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
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
            if self.mode == "mcdropout":
                h_prime = F.dropout(h_prime, p=0.3, training=True)
            
        # Classification
        x = self.classifier(h)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 8)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
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

    def __init__(self, params, hyperparams) -> None:
        super(M12, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.activation = torch.nn.LeakyReLU()

        in_channels = params["num_node_features"]
        out_channels = hyperparams["size_conv_layers"]

        for _ in range(hyperparams["num_conv_layers"]):
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

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
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
            if self.mode == "mcdropout":
                h = F.dropout(h, p=0.3, training=True)

        # Classification
        x = self.classifier(h)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 8)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M13(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + layer memory
    """

    def __init__(self, params, hyperparams) -> None:
        super(M13, self).__init__()

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
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        in_channels = (
            params["num_node_features"] + in_channels * hyperparams["num_conv_layers"]
        )
        mid_channels = hyperparams["size_final_mlp_layers"]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(mid_channels, 1),
        )

    def forward(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Convolutions
        x = data.x
        xs = [x]
        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)
            if i != len(self.conv) - 1:
                x = batch_norm(x)
                x = self.activation(x)
            xs.append(x)
            if self.mode == "mcdropout":
                x = F.dropout(x, p=0.3, training=True)

        # Concatenate final embedding and pooled representation
        x = torch.cat(xs, dim=1)

        # Classification
        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            learning_rate=learning_rate,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams
