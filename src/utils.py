import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

from collections import deque

from sklearn.metrics import auc, jaccard_score, matthews_corrcoef, \
    precision_score, recall_score, roc_auc_score, roc_curve, \
        ConfusionMatrixDisplay, PrecisionRecallDisplay


def average(lst):
    return sum(lst) / len(lst)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, delta):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Improvement tolerance.   
        """
        self.patience = patience
        self.delta = delta
        self.memory = deque(maxlen=10)
        self.first_run = True
        self.avg_old = None
        self.avg_new = None
        self.interval = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, criterion):
        if len(self.memory) < 10:
            self.memory.append(criterion)
        else:
            if self.first_run:
                self.avg_old = average(list(self.memory))
                self.first_run = False
            self.memory.pop()
            self.memory.append(criterion)
            self.interval += 1
            if self.interval == 10:
                self.avg_new = average(list(self.memory))
                diff = self.avg_new - self.avg_old
                self.avg_old = self.avg_new
                self.interval = 0
                if diff + self.delta > 0:
                    self.counter += 1
                    if self.counter >= self.patience: self.early_stop = True
                else:
                    self.counter = 0


def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def plot_losses(train_loss, val_loss, path):
    plt.figure()
    plt.plot(np.arange(0, len(train_loss), 1), train_loss, linestyle='-', linewidth=1, color ='orange', label='Training Loss')
    plt.plot(np.arange(0, len(train_loss), 1), val_loss, linestyle='-', linewidth=1, color ='blue', label='Validation Loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)


def plot_roc_curve(y_true, y_pred, save_plot, **path):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    roc_auc = auc(fpr, tpr)
    if save_plot:
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
        plt.savefig(path['path'])
    return best_threshold


def save_individual_results(output_directory, output_subdirectory, results_file_name, timestamp, \
    data_name, model_name, h_dim, dropout, num_heads, neg_slope, epochs, lr, wd, batch_size, oversampling, \
        y_pred, mol_id, y_true):

    # Compute top1 accuracy
    pred_top1 = []
    for id in np.unique(mol_id):
        mask = id == mol_id
        idx = np.argpartition(y_pred[mask], -1)[-1:]
        if y_true[mask][idx[0]]:
            pred_top1.append(1)
        else:
            pred_top1.append(0)
    top1 = np.sum(pred_top1)/len(np.unique(mol_id))

    # Compute top2 accuracy
    pred_top2 = []
    for id in np.unique(mol_id):
        mask = id == mol_id
        idx = np.argpartition(y_pred[mask], -2)[-2:]
        if y_true[mask][idx[0]] or y_true[mask][idx[1]]:
            pred_top2.append(1)
        else:
            pred_top2.append(0)
    top2 = np.sum(pred_top2)/len(np.unique(mol_id))

    # Compute and plot precision/recall curve
    PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    plt.savefig(os.path.join(output_subdirectory, 'pr_curve.png'))

    # Compute and plot ROC-AUC score and ROC-curve, get best threshold
    roc_auc = roc_auc_score(y_true, y_pred)
    best_threshold = plot_roc_curve(y_true, y_pred, True, path=os.path.join(output_subdirectory, 'roc_curve.png'))

    # Compute binary predictions from probability predictions with best threshold
    y_pred = ((y_pred > best_threshold))

    # Compute and plot confusion matrix 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(os.path.join(output_subdirectory, 'cm.png'))

    # Compute metrics that require binary predictions
    mcc = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)

    data = [timestamp, data_name, model_name, h_dim, dropout, num_heads, neg_slope, epochs, lr, wd, batch_size, oversampling, \
                round(best_threshold,3), round(mcc,3), round(top1,3), round(top2,3), \
                    round(jaccard,3), round(precision,3), round(recall,3), round(roc_auc,3)]
    if os.path.isfile(os.path.join(output_directory, results_file_name)) == False:
        with open(os.path.join(output_directory, results_file_name), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Subdirectory", "Data", "Model", "Dimension of Hidden Layers", "Dropout", "Heads", "Negative Slope", \
                "Training Epochs","Learning Rate","Weight Decay", "Batch Size", "Oversampling", "Optimal Threshold", "MCC", \
                    "Top1 Accuracy", "Top2 Accuracy", "Jaccard Score", "Precision", "Recall", "ROC AUC Score"])
    with open(os.path.join(output_directory, results_file_name), 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def save_average_results(output_directory, results_file_name, runs, data_name, model_name, h_dim, dropout, num_heads, \
    neg_slope, lr, wd, batch_size, oversampling, y_preds, mol_ids, y_trues):

    top1_list = []
    top2_list = []
    roc_auc_list = []
    best_threshold_list = []
    mcc_list = []
    jaccard_list = []
    precision_list = []
    recall_list = []

    for key in y_preds:

        # Compute top1 accuracy
        pred_top1 = []
        for id in np.unique(mol_ids[key]):
            mask = id == mol_ids[key]
            idx = np.argpartition(y_preds[key][mask], -1)[-1:]
            if y_trues[key][mask][idx[0]]:
                pred_top1.append(1)
            else:
                pred_top1.append(0)
        top1 = np.sum(pred_top1)/len(np.unique(mol_ids[key]))
        top1_list.append(top1)

        # Compute top2 accuracy
        pred_top2 = []
        for id in np.unique(mol_ids[key]):
            mask = id == mol_ids[key]
            idx = np.argpartition(y_preds[key][mask], -2)[-2:]
            if y_trues[key][mask][idx[0]] or y_trues[key][mask][idx[1]]:
                pred_top2.append(1)
            else:
                pred_top2.append(0)
        top2 = np.sum(pred_top2)/len(np.unique(mol_ids[key]))
        top2_list.append(top2)


        # Compute and plot ROC-AUC score and ROC-curve, get best threshold
        roc_auc = roc_auc_score(y_trues[key], y_preds[key])
        roc_auc_list.append(roc_auc)
        best_threshold = plot_roc_curve(y_trues[key], y_preds[key], False)
        best_threshold_list.append(best_threshold)

        # Compute binary predictions from probability predictions with best threshold
        y_preds[key] = ((y_preds[key] > best_threshold))


        # Compute metrics that require binary predictions
        mcc = matthews_corrcoef(y_trues[key], y_preds[key])
        mcc_list.append(mcc)
        jaccard = jaccard_score(y_trues[key], y_preds[key])
        jaccard_list.append(jaccard)
        precision = precision_score(y_trues[key], y_preds[key], zero_division=0)
        precision_list.append(precision)
        recall = recall_score(y_trues[key], y_preds[key])
        recall_list.append(recall)

    top1_mean = average(top1_list)
    top2_mean = average(top2_list)
    roc_auc_mean = average(roc_auc_list)
    best_threshold_mean = average(best_threshold_list)
    mcc_mean = average(mcc_list)
    jaccard_mean = average(jaccard_list)
    precision_mean = average(precision_list)
    recall_mean = average(recall_list)
    
    data = [runs, data_name, model_name, h_dim, dropout, num_heads, neg_slope, lr, wd, batch_size, oversampling, \
                round(best_threshold_mean,3), round(mcc_mean,3), round(top1_mean,3), round(top2_mean,3), \
                            round(jaccard_mean,3), round(precision_mean,3), round(recall_mean,3), round(roc_auc_mean,3)]
        
    if os.path.isfile(os.path.join(output_directory, results_file_name)) == False:
        with open(os.path.join(output_directory, results_file_name), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Runs", "Data", "Model", "Dimension of Hidden Layers", "Dropout", "Heads", "Negative Slope", \
                "Learning Rate","Weight Decay", "Batch Size", "Oversampling", "Optimal Threshold", "MCC", \
                    "Top1 Accuracy", "Top2 Accuracy", "Jaccard Score", "Precision", "Recall", "ROC AUC Score"])
    with open(os.path.join(output_directory, results_file_name), 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)