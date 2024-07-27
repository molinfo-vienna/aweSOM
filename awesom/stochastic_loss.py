import torch


# class StochasticLoss(torch.nn.Module):
#     """
#     Calculates the stochastic loss.

#     Args:
#         logits (torch.Tensor):  predictions from the model (logits)
#         stddevs (torch.Tensor):  predictions from the model (standard deviations)
#         targets (torch.Tensor): ground truth
#     """

#     def __init__(self, reduction="mean"):
#         super(StochasticLoss, self).__init__()
#         self.loss = torch.nn.BCELoss(reduction=reduction)
#         self.num_samples = 50

#     def forward(self, logits, stddevs, targets):
#         corrupted_logits = torch.cat(
#             [
#                 (logits + torch.mul(stddevs, torch.randn_like(logits))).reshape(-1, 1)
#                 for _ in range(self.num_samples)
#             ],
#             dim=1,
#         )
#         corrupted_probabilities = torch.sigmoid(corrupted_logits)
#         average_probabilities = torch.mean(corrupted_probabilities, dim=1)
#         return self.loss(average_probabilities, targets)


class StochasticLoss(torch.nn.Module):
    """
    Calculates the stochastic loss.

    Args:
        logits (torch.Tensor):  predictions from the model (logits)
        stddevs (torch.Tensor):  predictions from the model (standard deviations)
        targets (torch.Tensor): ground truth
    """

    def __init__(self):
        super(StochasticLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.num_samples = 50

    def forward(self, logits, stddevs, targets):
        corrupted_logits = torch.cat(
            [
                (logits + torch.mul(stddevs, torch.randn_like(logits))).reshape(-1, 1)
                for _ in range(self.num_samples)
            ],
            dim=1,
        )
        expanded_targets = targets.unsqueeze(1).repeat(1, self.num_samples)
        stochastic_loss = self.loss(corrupted_logits, expanded_targets)
        stochastic_loss = torch.mean(stochastic_loss, dim=1)
        stochastic_loss = torch.sum(stochastic_loss)
        return stochastic_loss
    