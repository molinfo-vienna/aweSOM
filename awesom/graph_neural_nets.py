import torch
from torch_geometric.nn import (
    GATv2Conv,
    GINConv,
    GINEConv,
    MFConv,
    TransformerConv,
    global_add_pool,
)
from typing import Any, List, Optional, Tuple, Union


__all__ = ["GATv2", "GIN", "GINNA", "GINPlus", "GINE", "GINENA", "GINEPlus", "MF", "TF"]


class GATv2(torch.nn.Module):
    """The GATv2 operator from the “How Attentive are Graph Attention Networks?” paper,
    which fixes the static attention problem of the standard GATConv layer.
    Since the linear layers in the standard GAT are applied right after each other,
    the ranking of attended nodes is unconditioned on the query node.
    In contrast, in GATv2, every node can attend to any other node.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int,
        negative_slope: float,
        dropout: float,
        n_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(GATv2, self).__init__()

        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        self.conv = torch.nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=out_channels * heads,
                    out_channels=out_channels,
                    heads=heads,
                    negative_slope=negative_slope,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(
            out_channels * (n_conv_layers + 2) * heads, size_classify_layers
        )

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)
        hs.append(h)

        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class GIN(torch.nn.Module):
    """The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        n_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.BatchNorm1d(out_channels, out_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
            ),
            train_eps=True,
        )

        self.conv = torch.nn.ModuleList(
            [
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(out_channels, out_channels),
                        torch.nn.BatchNorm1d(out_channels, out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Dropout(p=dropout),
                    ),
                    train_eps=True,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(
            out_channels * (n_conv_layers + 2), size_classify_layers
        )

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        h = self.conv1(x, edge_index)
        h = self.leaky_relu(h)
        hs.append(h)

        for layer in self.conv:
            h = layer(h, edge_index)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class GINNA(torch.nn.Module):
    """The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper,
    without aggregation over the intermediate layers, and without pooling over the molecule.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        n_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(GINNA, self).__init__()

        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.BatchNorm1d(out_channels, out_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
            ),
            train_eps=True,
        )

        self.conv = torch.nn.ModuleList(
            [
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(out_channels, out_channels),
                        torch.nn.BatchNorm1d(out_channels, out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Dropout(p=dropout),
                    ),
                    train_eps=True,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(out_channels, size_classify_layers)

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        h = self.conv1(x, edge_index)
        h = self.leaky_relu(h)

        for layer in self.conv:
            h = layer(h, edge_index)
            h = self.leaky_relu(h)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class GINPlus(torch.nn.Module):
    """The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper.
    Difference to GIN: GINPlus can be be Optuna optimized over the depth of the neural nets in the convolutional layers.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        n_conv_layers: int,
        depth_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(GINPlus, self).__init__()

        first_conv_internal_layers: List[Any] = []
        for _ in range(depth_conv_layers - 1):
            first_conv_internal_layers.append(
                torch.nn.Linear(out_channels, out_channels)
            )
            first_conv_internal_layers.append(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(out_channels, out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=dropout),
                )
            )

        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.BatchNorm1d(out_channels, out_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
                *first_conv_internal_layers
            ),
            train_eps=True,
        )

        other_conv_internal_layers: List[Any] = []
        for _ in range(depth_conv_layers):
            other_conv_internal_layers.append(
                torch.nn.Linear(out_channels, out_channels)
            )
            other_conv_internal_layers.append(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(out_channels, out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=dropout),
                )
            )

        self.conv = torch.nn.ModuleList(
            [
                GINConv(
                    torch.nn.Sequential(*other_conv_internal_layers),
                    train_eps=True,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(
            out_channels * (n_conv_layers + 2), size_classify_layers
        )

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        h = self.conv1(x, edge_index)
        h = self.leaky_relu(h)
        hs.append(h)

        for layer in self.conv:
            h = layer(h, edge_index)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class GINE(torch.nn.Module):
    """The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float,
        n_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(GINE, self).__init__()

        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.BatchNorm1d(out_channels, out_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
            ),
            train_eps=True,
            edge_dim=edge_dim,
        )

        self.conv = torch.nn.ModuleList(
            [
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(out_channels, out_channels),
                        torch.nn.BatchNorm1d(out_channels, out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Dropout(p=dropout),
                    ),
                    train_eps=True,
                    edge_dim=edge_dim,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(
            out_channels * (n_conv_layers + 2), size_classify_layers
        )

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)
        hs.append(h)

        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class GINENA(torch.nn.Module):
    """The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper,
    without aggregation over the intermediate layers, and without pooling over the molecule.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float,
        n_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(GINENA, self).__init__()

        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.BatchNorm1d(out_channels, out_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
            ),
            train_eps=True,
            edge_dim=edge_dim,
        )

        self.conv = torch.nn.ModuleList(
            [
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(out_channels, out_channels),
                        torch.nn.BatchNorm1d(out_channels, out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Dropout(p=dropout),
                    ),
                    train_eps=True,
                    edge_dim=edge_dim,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(out_channels, size_classify_layers)

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)

        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class GINEPlus(torch.nn.Module):
    """The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    Difference to GINE: GINEPlus can be be Optuna optimized over the depth of the neural nets in the convolutional layers.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float,
        n_conv_layers: int,
        depth_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(GINEPlus, self).__init__()

        first_conv_internal_layers: List[Any] = []
        for _ in range(depth_conv_layers - 1):
            first_conv_internal_layers.append(
                torch.nn.Linear(out_channels, out_channels)
            )
            first_conv_internal_layers.append(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(out_channels, out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=dropout),
                )
            )

        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.BatchNorm1d(out_channels, out_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
                *first_conv_internal_layers
            ),
            train_eps=True,
            edge_dim=edge_dim,
        )

        other_conv_internal_layers: List[Any] = []
        for _ in range(depth_conv_layers):
            other_conv_internal_layers.append(
                torch.nn.Linear(out_channels, out_channels)
            )
            other_conv_internal_layers.append(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(out_channels, out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=dropout),
                )
            )

        self.conv = torch.nn.ModuleList(
            [
                GINEConv(
                    torch.nn.Sequential(*other_conv_internal_layers),
                    train_eps=True,
                    edge_dim=edge_dim,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(
            out_channels * (n_conv_layers + 2), size_classify_layers
        )

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)
        hs.append(h)

        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class MF(torch.nn.Module):
    """The graph neural network operator from the “Convolutional Networks on Graphs for Learning Molecular Fingerprints” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_degree: int,
        n_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(MF, self).__init__()

        self.conv1 = MFConv(
            in_channels=in_channels, out_channels=out_channels, max_degree=max_degree
        )

        self.conv = torch.nn.ModuleList(
            [
                MFConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    max_degree=max_degree,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(
            out_channels * (n_conv_layers + 2), size_classify_layers
        )

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        h = self.conv1(x, edge_index)
        h = self.leaky_relu(h)
        hs.append(h)

        for layer in self.conv:
            h = layer(h, edge_index)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)


class TF(torch.nn.Module):
    """The graph transformer operator from the “Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html#torch_geometric.nn.conv.TransformerConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int,
        dropout: float,
        n_conv_layers: int,
        n_classifier_layers: int,
        size_classify_layers: int,
    ) -> None:
        super(TF, self).__init__()

        self.conv1 = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=True,
            beta=False,
            dropout=dropout,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True,
        )

        self.conv = torch.nn.ModuleList(
            [
                TransformerConv(
                    in_channels=out_channels * heads,
                    out_channels=out_channels,
                    heads=heads,
                    concat=True,
                    beta=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    bias=True,
                    root_weight=True,
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.classifier1 = torch.nn.Linear(
            out_channels * heads * (n_conv_layers + 2), size_classify_layers
        )

        self.classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(size_classify_layers, size_classify_layers)
                for _ in range(n_classifier_layers)
            ]
        )

        self.final = torch.nn.Linear(size_classify_layers, 1)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forwar(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Compute node intermediate embeddings
        hs = []
        h = self.conv1(x, edge_index, edge_attr)
        h = self.leaky_relu(h)
        hs.append(h)

        for layer in self.conv:
            h = layer(h, edge_index, edge_attr)
            h = self.leaky_relu(h)
            hs.append(h)

        # Pooling
        h_pool = global_add_pool(h, batch)
        num_atoms_per_mol = torch.unique(batch, sorted=False, return_counts=True)[1]
        h_pool_ = torch.repeat_interleave(h_pool, num_atoms_per_mol, dim=0)

        # Concatenate embeddings
        h = torch.cat((*hs, h_pool_), dim=1)

        # Classify
        h = self.classifier1(h)
        for layer in self.classifier:
            h = layer(h)
            h = self.leaky_relu(h)
        h = self.final(h)

        return torch.sigmoid(h)
