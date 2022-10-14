import logging
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time criterion improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.       
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.criterion_opt = None

    def __call__(self, criterion, opt_mode, model):
        if opt_mode == 'min':
            score = -criterion
            self.criterion_opt = np.Inf
        elif opt_mode == 'max':
            score = criterion
            self.criterion_opt = -np.Inf
        else:
            logging.warning("Unsupported optimizing criterion.")
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(criterion, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(criterion, model)
            self.counter = 0

    def save_checkpoint(self, criterion, model):
        torch.save(model.state_dict(), self.path)
        self.criterion_opt = criterion