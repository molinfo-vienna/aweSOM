import logging
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, delta=0.001):
        """
        Args:
            patience (int): How long to wait after last time criterion improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.   
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, criterion, opt_mode):
        if opt_mode == 'min':
            score = -criterion
        elif opt_mode == 'max':
            score = criterion
        else:
            logging.warning("Unsupported optimizing criterion.")

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_score = score