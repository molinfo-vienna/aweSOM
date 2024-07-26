import torch


class StochasticLoss(torch.nn.Module):
    """
    Calculates the stochastic loss.

    Args:
        logits (torch.Tensor):  predictions from the model (logits)
        stddevs (torch.Tensor):  predictions from the model (standard deviations)
        targets (torch.Tensor): ground truth
    """

    def __init__(self, reduction="mean"):
        super(StochasticLoss, self).__init__()
        self.loss = torch.nn.BCELoss(reduction=reduction)
        self.num_samples = 100

    def forward(self, logits, stddevs, targets):
        corrupted_logits = torch.cat(
            [
                (logits + torch.mul(stddevs, torch.randn_like(logits))).reshape(-1, 1)
                for _ in range(self.num_samples)
            ],
            dim=1,
        )
        corrupted_probabilities = torch.sigmoid(corrupted_logits)
        average_probabilities = torch.mean(corrupted_probabilities, dim=1)
        return self.loss(average_probabilities, targets)
