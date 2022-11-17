import csv
import logging
import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc, jaccard_score, matthews_corrcoef, \
    precision_score, recall_score, roc_auc_score, roc_curve, \
        ConfusionMatrixDisplay, PrecisionRecallDisplay
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, delta):
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


def plot_losses(train_loss, val_loss, path):
    plt.plot(np.arange(0, len(train_loss), 1), train_loss, linestyle='-', linewidth=1, color ='orange', label='Training Loss')
    plt.plot(np.arange(0, len(train_loss), 1), val_loss, linestyle='-', linewidth=1, color ='blue', label='Validation Loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)


def plot_roc_curve(y_true, y_pred, path):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
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
    plt.savefig(path)
    return best_threshold

def save_evaluation_results(output_directory, output_subdirectory, timestamp, \
    data_name, model_name, h_dim, num_heads, epochs, lr, wd, \
        val_pred, val_mol_ids, val_true):
    # Compute top1 accuracy
    pred_top1 = []
    for id in np.unique(val_mol_ids):
        mask = id == val_mol_ids
        idx = np.argpartition(val_pred[mask][:,0], -1)[-1:]
        if val_true[mask][idx[0]]:
            pred_top1.append(1)
        else:
            pred_top1.append(0)
    val_acc1 = np.sum(pred_top1)/len(np.unique(val_mol_ids))

    # Compute top2 accuracy
    pred_top2 = []
    for id in np.unique(val_mol_ids):
        mask = id == val_mol_ids
        idx = np.argpartition(val_pred[mask][:,0], -2)[-2:]
        if val_true[mask][idx[0]] or val_true[mask][idx[1]]:
            pred_top2.append(1)
        else:
            pred_top2.append(0)
    val_acc2 = np.sum(pred_top2)/len(np.unique(val_mol_ids))

    # Compute top3 accuracy
    pred_top3 = []
    for id in np.unique(val_mol_ids):
        mask = id == val_mol_ids
        idx = np.argpartition(val_pred[mask][:,0], -3)[-3:]
        if val_true[mask][idx[0]] or val_true[mask][idx[1]] or val_true[mask][idx[2]]:
            pred_top3.append(1)
        else:
            pred_top3.append(0)
    val_acc3 = np.sum(pred_top3)/len(np.unique(val_mol_ids))

    # Compute and plot precision/recall curve
    PrecisionRecallDisplay.from_predictions(val_true, val_pred)
    plt.savefig(os.path.join(output_subdirectory, 'pr_curve.png'))

    # Compute and plot ROC-AUC score and ROC-curve, get best threshold
    val_roc_auc = roc_auc_score(val_true, val_pred)
    best_threshold = plot_roc_curve(val_true, val_pred, os.path.join(output_subdirectory, 'roc_curve.png'))

    # Compute binary predictions from probability predictions with best threshold
    val_pred = ((val_pred > best_threshold)[:,0])

    # Compute and plot confusion matrix 
    ConfusionMatrixDisplay.from_predictions(val_true, val_pred)
    plt.savefig(os.path.join(output_subdirectory, 'cm.png'))

    # Compute metrics that require binary predictions
    val_mcc = matthews_corrcoef(val_true, val_pred)
    val_jacc = jaccard_score(val_true, val_pred)
    val_prec = precision_score(val_true, val_pred, zero_division=0)
    val_rec = recall_score(val_true, val_pred)

    data = [timestamp, data_name, model_name, h_dim, num_heads, epochs, lr, wd, \
        best_threshold, val_mcc, val_acc1, val_acc2, val_acc3, val_jacc, val_prec, val_rec, val_roc_auc]
    with open(os.path.join(output_directory, "results.csv"), 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
