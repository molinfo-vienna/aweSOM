from typing import Union

import optuna
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    BatchNorm,
    GINEConv,
    global_add_pool,
)


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
