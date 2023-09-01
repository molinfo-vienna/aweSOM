import torch
from torch_geometric.nn import (
    GATv2Conv,
    GINConv,
    GINEConv,
    global_add_pool,
)
from typing import List, Optional, Tuple, Union


__all__ = ["GATv2", "GIN", "GINE"]


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
        edge_dim: int,
        heads: int,
        negative_slope: float,
        dropout: float,
        size_conv_layers: List[int],
        size_classify_layers: List[int],
    ) -> None:
        super(GATv2, self).__init__()

        self.conv = torch.nn.ModuleList()
        for out_channels in size_conv_layers:
            self.conv.append(
                GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                negative_slope=negative_slope,
                dropout=dropout,
                edge_dim=edge_dim,
                )
            )
            in_channels = out_channels * heads

        self.classifier = torch.nn.ModuleList()
        in_channels = (sum(size_conv_layers) + size_conv_layers[-1]) * heads
        for out_channels in size_classify_layers:
            self.classifier.append(
                torch.nn.Linear(in_channels, out_channels)
            )
            in_channels = out_channels

        self.final = torch.nn.Linear(in_channels, 1)

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
        for layer in self.conv:
            h = layer(x, edge_index, edge_attr)
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


class GIN(torch.nn.Module):
    """The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv
    """

    def __init__(
        self,
        in_channels: int,
        dropout: float,
        size_conv_layers: List[int],
        size_classify_layers: List[int],
    ) -> None:
        super(GIN, self).__init__()

        self.conv = torch.nn.ModuleList()
        for out_channels in size_conv_layers:
            self.conv.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        torch.nn.BatchNorm1d(out_channels, out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Dropout(p=dropout),
                    ),
                    train_eps=True,
                )
            )
            in_channels = out_channels

        self.classifier = torch.nn.ModuleList()
        in_channels = sum(size_conv_layers) + size_conv_layers[-1]
        for out_channels in size_classify_layers:
            self.classifier.append(
                torch.nn.Linear(in_channels, out_channels)
            )
            in_channels = out_channels

        self.final = torch.nn.Linear(in_channels, 1)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(
            self,
            x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            edge_index: torch.Tensor,
            batch: Optional[List[int]] = None,
        ) -> torch.Tensor:
            # Compute node intermediate embeddings
            hs = []
            for layer in self.conv:
                h = layer(x, edge_index)
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


class GINE(torch.nn.Module):
    """The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        dropout: float,
        size_conv_layers: List[int],
        size_classify_layers: List[int],
    ) -> None:
        super(GINE, self).__init__()

        self.conv = torch.nn.ModuleList()
        for out_channels in size_conv_layers:
            self.conv.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_channels),
                        torch.nn.BatchNorm1d(out_channels, out_channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.Dropout(p=dropout),
                    ),
                    train_eps=True,
                    edge_dim=edge_dim,
                )
            )
            in_channels = out_channels

        self.classifier = torch.nn.ModuleList()
        in_channels = sum(size_conv_layers) + size_conv_layers[-1]
        for out_channels in size_classify_layers:
            self.classifier.append(
                torch.nn.Linear(in_channels, out_channels)
            )
            in_channels = out_channels

        self.final = torch.nn.Linear(in_channels, 1)

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
        for layer in self.conv:
            h = layer(x, edge_index, edge_attr)
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
    