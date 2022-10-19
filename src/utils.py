import logging
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

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
            if self.best_score is None: self.best_score = criterion
            elif criterion >= self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience: self.early_stop = True
            else:
                self.counter = 0
                self.best_score = criterion
        elif opt_mode == 'max':
            if self.best_score is None: self.best_score = criterion
            elif criterion <= self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience: self.early_stop = True
            else:
                self.counter = 0
                self.best_score = criterion
        else:
            logging.warning("Unsupported optimizing criterion.")


def roc_curve_display(y_true, y_pred, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve, Validation Set")
    plt.legend(loc="lower right")
    plt.savefig(output_path)