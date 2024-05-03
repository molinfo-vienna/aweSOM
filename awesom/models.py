import torch
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
from lightning import LightningModule
from torchmetrics import AUROC, MatthewsCorrCoef


class MCCLoss(torch.nn.Module):
    """
    Calculates the Matthews Correlation Coefficient loss.

    Args:
        outputs (torch.Tensor): predictions
        targets (torch.Tensor): ground truth
    """

    def __init__(self):
        super(MCCLoss, self).__init__()

    def forward(self, outputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(outputs, targets))
        tn = torch.sum(torch.mul((1 - outputs), (1 - targets)))
        fp = torch.sum(torch.mul(outputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - outputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc
    

class HeteroscedasticLoss(torch.nn.Module):
    """
    Calculates the heteroscedastic classification loss according to Kendall and Gal (2017).
    https://arxiv.org/abs/1703.04977

    Args:
        outputs (torch.Tensor): predictions (mean and standard deviation)
        targets (torch.Tensor): ground truth
    """

    def __init__(self, pos_weight):
        super(HeteroscedasticLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, outputs, targets):
        """
        Computes the heteroscedastic loss by:
        1) corrupting the logits with Gaussian noise with variance sigma^2,
        2) calculating the negative log likelihood of the corrupted logits,
        3) averaging the negative log likelihood over the batch.
        """
        mu = outputs[:, 0]
        sigma = outputs[:, 1]
        loss = 0
        for _ in range(10):
            noise = torch.randn_like(mu) * sigma
            logits = mu + noise
            loss += torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum", pos_weight=self.pos_weight)
        return loss


class GNN(LightningModule):
    """
    Parent class for the graph neural network models.

    Args:
        params (dict): dictionary of parameters
        hyperparams (dict): dictionary of hyperparameters
        architecture (str): model architecture
        pos_weight (float): positive class weight for the loss function
    """
    def __init__(
        self,
        params,
        hyperparams,
        architecture,
        pos_weight,
    ) -> None:
        super(GNN, self).__init__()

        self.save_hyperparameters()

        model_dict = {
            "M1": M1,
            "M2": M2,
            "M4": M4,
            "M6": M6,
            "M7": M7,
            "M8": M8,
            "M9": M9,
            "M11": M11,
            "M12": M12,
            "M13": M13,
            "M14": M14,
        }

        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float).cuda()
        # self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="sum")
        self.loss_function = MCCLoss()
        # self.loss_function = HeteroscedasticLoss(pos_weight=self.pos_weight)

        self.model = model_dict[architecture](params, hyperparams, self.pos_weight)

        self.learning_rate = hyperparams["learning_rate"]
        self.weight_decay = hyperparams["weight_decay"]

        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.train_mcc = MatthewsCorrCoef(task="binary")
        self.val_mcc = MatthewsCorrCoef(task="binary")

    def configure_optimizers(self):
        """Configures the optimizers and learning rate scheduler.

        Creates an AdamW optimizer with the given learning rate and weight decay.
        Also creates a ReduceLROnPlateau learning rate scheduler that reduces the
        learning rate by a factor of 0.1 when the validation loss plateaus.

        Returns:
            A dictionary with the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay,
            amsgrad=False,
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
    
    # def step(self, batch):
    #     """Step function for training with BCEWithLogitsLoss.

    #     Args:
    #         batch (torch_geometric.data.Data): batch of data

    #     Returns:
    #         loss (torch.Tensor): loss value
    #         logits (torch.Tensor): model output
    #     """
    #     logits = self.model(batch)
    #     loss = self.loss_function(logits, batch.y.float())
    #     return loss, logits

    def step(self, batch):
        """Step function for training with MCCLoss.

        Args:
            batch (torch_geometric.data.Data): batch of data

        Returns:
            loss (torch.Tensor): loss value
            logits (torch.Tensor): model output
        """
        logits = self.model(batch)
        loss = self.loss_function(torch.sigmoid(logits), batch.y.float())
        return loss, logits

    # def step(self, batch):
    #     """Step function for training with heteroscedastic loss.

    #     Args:
    #         batch (torch_geometric.data.Data): batch of data

    #     Returns:
    #         loss (torch.Tensor): loss value
    #         outputs (torch.Tensor): model outputs (mean and standard deviation)
    #     """
    #     outputs = self.model(batch)
    #     loss = self.loss_function(outputs, batch.y.float())
    #     return loss, outputs

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams,
            {
                "train/loss": 1,
                "train/auroc": 0,
                "train/mcc": 0,
                "val/loss": 1,
                "val/auroc": 0,
                "val/mcc": 0,
            },
        )

    def training_step(self, batch, batch_idx):
        loss, logits = self.step(batch)
        # loss, outputs = self.step(batch)
        # logits, sigma = outputs[:, 0], outputs[:, 1]
        y_hats = torch.sigmoid(logits)
        self.train_auroc(y_hats, batch.y)
        self.train_mcc(y_hats, batch.y)
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
        loss, logits = self.step(batch)
        # loss, outputs = self.step(batch)
        # logits, sigma = outputs[:, 0], outputs[:, 1]
        y_hats = torch.sigmoid(logits)
        self.val_auroc(y_hats, batch.y)
        self.val_mcc(y_hats, batch.y)
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
        """Predict step function for inference with BCEWithLogitsLoss or MCCLoss.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        _, logits = self.step(batch)
        return logits, batch.y, batch.mol_id, batch.atom_id

    # def predict_step(self, batch, batch_idx):
    #     """Predict step function for inference with heteroscedastic loss.

    #     Args:
    #         batch (_type_): _description_
    #         batch_idx (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     _, outputs = self.step(batch)
    #     logits, sigma = outputs[:, 0], outputs[:, 1]
    #     return logits, sigma, batch.y, batch.mol_id, batch.atom_id


class M1(torch.nn.Module):
    """
    The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper.
    https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.conv.GINConv.html
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
        super(M1, self).__init__()

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
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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
                x = self.activation(x)

        # Classification
        x = self.final(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
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

    def __init__(self, params, hyperparams, pos_weight) -> None:
        super(M2, self).__init__()

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

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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
                x = self.activation(x)

        # Classification
        x = self.final(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M4(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Pooling for context
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
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
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
            torch.nn.Linear(mid_channels, 1),
        )

        self.logits_layer = torch.nn.Linear(in_channels*2, 1)
        self.sigma_layer = torch.nn.Linear(in_channels*2, 1)

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
                x = self.activation(x)

        # Pooling for context
        x_pool = global_add_pool(x, data.batch)
        num_atoms_per_mol = torch.unique(data.batch, sorted=False, return_counts=True)[
            1
        ]
        x_pool_expanded = torch.repeat_interleave(x_pool, num_atoms_per_mol, dim=0)

        # Concatenate final embedding and pooled representation
        x = torch.cat((x, x_pool_expanded), dim=1)

        # Classification with BCEWithLogitsLoss or MCCLoss
        x = self.final(x)
        return torch.flatten(x)

        # # Classification with heteroscedastic loss
        # logits = self.logits_layer(x)
        # sigma = self.sigma_layer(x)
        # output = torch.cat((logits, sigma), dim=1)
        # return output

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M6(torch.nn.Module):
    """The GATv2 operator from the “How Attentive are Graph Attention Networks?” paper,
    which fixes the static attention problem of the standard GATConv layer.
    Since the linear layers in the standard GAT are applied right after each other,
    the ranking of attended nodes is unconditioned on the query node.
    In contrast, in GATv2, every node can attend to any other node.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
        super(M6, self).__init__()

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

        mid_channels = hyperparams["size_final_mlp_layers"]
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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
                x = self.activation(x)

        # Classification
        x = self.final(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)
        heads = trial.suggest_int("heads", 1, 4)
        negative_slope = trial.suggest_float("negative_slope", 0.1, 0.9)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
            heads=heads,
            negative_slope=negative_slope,
        )

        return hyperparams


class M7(torch.nn.Module):
    """The graph neural network operator from the “Convolutional Networks on Graphs for Learning Molecular Fingerprints” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
        super(M7, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.activation = torch.nn.LeakyReLU()

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
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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
                x = self.activation(x)

        # Classification
        x = self.final(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        max_degree = trial.suggest_int("max_degree", 1, 6)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            max_degree=max_degree,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M8(torch.nn.Module):
    """The chebyshev spectral graph convolutional operator from the
    “Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ChebConv.html
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
        super(M8, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.activation = torch.nn.LeakyReLU()

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
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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
                x = self.activation(x)

        # Classification
        x = self.final(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        filter_size = trial.suggest_int("filter_size", 1, 10)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            filter_size=filter_size,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M9(torch.nn.Module):
    """The chebyshev spectral graph convolutional operator from the
    “Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ChebConv.html
    + context pooling
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
        super(M9, self).__init__()

        self.conv = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.activation = torch.nn.LeakyReLU()

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
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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
                x = self.activation(x)

        # Pooling for context
        x_pool = global_add_pool(x, data.batch)
        num_atoms_per_mol = torch.unique(data.batch, sorted=False, return_counts=True)[
            1
        ]
        x_pool_expanded = torch.repeat_interleave(x_pool, num_atoms_per_mol, dim=0)

        # Concatenate final embedding and pooled representation
        x = torch.cat((x, x_pool_expanded), dim=1)

        # Classification
        x = self.final(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        filter_size = trial.suggest_int("filter_size", 1, 10)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            filter_size=filter_size,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M11(torch.nn.Module):
    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Skip connections in the style of DenseNet
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
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
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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

        # Classification
        h = self.final(h)

        return torch.flatten(h)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
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

    def __init__(self, params, hyperparams, pos_weight) -> None:
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
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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

        # Classification
        h = self.final(h)

        return torch.flatten(h)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M13(torch.nn.Module):

    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Pooling for context
    + Learned molecular features as additional input
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
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

        in_channels_mol = params["num_mol_features"]
        out_channels_mol = hyperparams["size_mol_mlp_layers"]
        self.mol_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels_mol, out_channels_mol),
            BatchNorm(out_channels_mol),
            self.activation,
            torch.nn.Linear(out_channels_mol, out_channels_mol),
        )

        in_channels = in_channels + out_channels_mol
        mid_channels = hyperparams["size_final_mlp_layers"]
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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

        # Compute molecular embeddings
        x_mol = self.mol_mlp(data.mol_x)

        # Concatenate atom embeddings and molecular embeddings
        x = torch.cat((x_atom, x_mol), dim=1)

        # Classification
        x = self.final(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_mol_mlp_layers = trial.suggest_int("size_mol_mlp_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_mol_mlp_layers=size_mol_mlp_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams


class M14(torch.nn.Module):

    """
    The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper.
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    + Pooling for context
    + Learned molecular features as additional input
    """

    def __init__(self, params, hyperparams, pos_weight) -> None:
        super(M14, self).__init__()

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

        in_channels_mol = params["num_mol_features"]
        out_channels_mol = hyperparams["size_mol_mlp_layers"]
        self.mol_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels_mol, out_channels_mol),
            BatchNorm(out_channels_mol),
            self.activation,
            torch.nn.Linear(out_channels_mol, out_channels_mol),
        )

        in_channels = (in_channels * 2) + out_channels_mol
        mid_channels = hyperparams["size_final_mlp_layers"]
        self.final = torch.nn.Sequential(
            torch.nn.Linear(in_channels, mid_channels),
            BatchNorm(mid_channels),
            self.activation,
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

        # Pooling for context
        x_pool = global_add_pool(x_atom, data.batch)
        num_atoms_per_mol = torch.unique(data.batch, sorted=False, return_counts=True)[
            1
        ]
        x_pool_expanded = torch.repeat_interleave(x_pool, num_atoms_per_mol, dim=0)

        # Concatenate final embedding and pooled representation
        x_atom = torch.cat((x_atom, x_pool_expanded), dim=1)

        # Compute molecular embeddings
        x_mol = self.mol_mlp(data.mol_x)

        # Concatenate atom embeddings and molecular embeddings
        x = torch.cat((x_atom, x_mol), dim=1)

        # Classification
        x = self.final(x)

        return torch.flatten(x)

    @classmethod
    def get_params(self, trial):
        batch_size = trial.suggest_int("batch_size", 16, 256)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        size_conv_layers = trial.suggest_int("size_conv_layers", 32, 512)
        size_mol_mlp_layers = trial.suggest_int("size_mol_mlp_layers", 32, 512)
        size_final_mlp_layers = trial.suggest_int("size_final_mlp_layers", 32, 512)

        hyperparams = dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            size_conv_layers=size_conv_layers,
            size_mol_mlp_layers=size_mol_mlp_layers,
            size_final_mlp_layers=size_final_mlp_layers,
        )

        return hyperparams
