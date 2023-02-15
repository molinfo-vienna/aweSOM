import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

from collections import Counter

import torch.nn.functional as F

from sklearn.metrics import (
    auc,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


################################
##################### NN related utility functions

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=1, delta=0):
        """
        Args:
            patience (int): number of epochs with no improvement of the validation or test loss after which training will be stopped
            delta (float): minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

################################
##################### loss utility functions

class MCC_Loss(torch.nn.Module):
    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        TP = torch.sum(predictions * targets)
        TN = torch.sum((1 - predictions) * (1 - targets))
        FP = torch.sum((1 - targets) * predictions)
        FN = torch.sum(targets * (1 - predictions))
        MCC_loss = 1 - (TP * TN - FP * FN) / (
            torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
        return MCC_loss


class MCC_BCE_Loss(torch.nn.Module):
    def __init__(self):
        super(MCC_BCE_Loss, self).__init__()

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        TP = torch.sum(predictions * targets)
        TN = torch.sum((1 - predictions) * (1 - targets))
        FP = torch.sum((1 - targets) * predictions)
        FN = torch.sum(targets * (1 - predictions))
        MCC_loss = 1 - (TP * TN - FP * FN) / (
            torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
        BCE_loss = F.binary_cross_entropy(predictions, targets, reduction="sum")
        MCC_BCE_loss = MCC_loss + BCE_loss
        return MCC_BCE_loss


################################
##################### general utility functions

def average(lst):
    return sum(lst) / len(lst)

def seed_everything(seed=42):
    """ "
    Seed everything.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def plot_losses(train_loss, val_loss, path):
    plt.figure()
    plt.plot(
        np.arange(0, len(train_loss), 1),
        train_loss,
        linestyle="-",
        linewidth=1,
        color="orange",
        label="Training Loss",
    )
    plt.plot(
        np.arange(0, len(train_loss), 1),
        val_loss,
        linestyle="-",
        linewidth=1,
        color="blue",
        label="Validation Loss",
    )
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
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
        plt.savefig(path["path"])
    return best_threshold


def save_individual(
    output_directory,
    output_subdirectory,
    fold_num,
    results_file_name,
    hdim,
    dropout,
    lr,
    wd,
    bs,
    final_num_epochs,
    val_loss,
    y_pred,
    y_true,
    mol_id,
    atom_id,
):

    # Compute top1 accuracy
    pred_top1 = []
    for id in np.unique(mol_id):
        mask = id == mol_id
        idx = np.argpartition(y_pred[mask], -1)[-1:]
        if y_true[mask][idx[0]]:
            pred_top1.append(1)
        else:
            pred_top1.append(0)
    top1 = np.sum(pred_top1) / len(np.unique(mol_id))

    # Compute top2 accuracy
    pred_top2 = []
    for id in np.unique(mol_id):
        mask = id == mol_id
        idx = np.argpartition(y_pred[mask], -2)[-2:]
        if y_true[mask][idx[0]] or y_true[mask][idx[1]]:
            pred_top2.append(1)
        else:
            pred_top2.append(0)
    top2 = np.sum(pred_top2) / len(np.unique(mol_id))

    # Compute and plot ROC-AUC score and ROC-curve, get best threshold
    roc_auc = roc_auc_score(y_true, y_pred)
    best_threshold = plot_roc_curve(y_true, y_pred, False)

    # Compute binary predictions from probability predictions with best threshold
    y_pred_bin = y_pred > best_threshold

    # Compute metrics that require binary predictions
    mcc = matthews_corrcoef(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin, zero_division=0)
    recall = recall_score(y_true, y_pred_bin)

    # Save molecular identifiers (mol_id), true labels (y_true), and predicted labels (y_pred) to a csv file:
    rows = zip(mol_id, atom_id, y_true, y_pred, y_pred_bin)
    with open(
        os.path.join(output_subdirectory, "predictions" + str(fold_num) + ".csv"),
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "mol_id",
                "atom_id",
                "true_label",
                "predicted_label",
                "predicted_binary_label",
            )
        )
        writer.writerows(rows)
    f.close()

    with open(
        os.path.join(output_subdirectory, "hp.csv"),
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(        
            [
                hdim,
                dropout,
                lr,
                wd,
                bs,
            ]
        )

    data = [
        str(output_subdirectory),
        hdim,
        dropout,
        lr,
        wd,
        bs,
        fold_num,
        final_num_epochs,
        val_loss,
        round(best_threshold, 3),
        round(mcc, 3),
        round(top1, 3),
        round(top2, 3),
        round(precision, 3),
        round(recall, 3),
        round(roc_auc, 3),
    ]

    if os.path.isfile(os.path.join(output_directory, results_file_name)) == False:
        with open(
            os.path.join(output_directory, results_file_name),
            "w",
            encoding="UTF8",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Results Folder",
                    "Dimension of Hidden Layers",
                    "Dropout",
                    "Learning Rate",
                    "Weight Decay",
                    "Batch Size",
                    "Fold",
                    "Epochs",
                    "Validation Loss",
                    "Optimal Threshold",
                    "MCC",
                    "Top1 Accuracy",
                    "Top2 Accuracy",
                    "Precision",
                    "Recall",
                    "ROC AUC Score",
                ]
            )
    with open(
        os.path.join(output_directory, results_file_name),
        "a",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(data)


def save_average(
    output_directory,
    results_file_name,
    hdim,
    dropout,
    lr,
    wd,
    bs,
    y_preds,
    y_trues,
    mol_ids,
):

    top1_list = []
    top2_list = []
    roc_auc_list = []
    best_threshold_list = []
    mcc_list = []
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
        top1 = np.sum(pred_top1) / len(np.unique(mol_ids[key]))
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
        top2 = np.sum(pred_top2) / len(np.unique(mol_ids[key]))
        top2_list.append(top2)

        # Compute and plot ROC-AUC score and ROC-curve, get best threshold
        roc_auc = roc_auc_score(y_trues[key], y_preds[key])
        roc_auc_list.append(roc_auc)
        best_threshold = plot_roc_curve(y_trues[key], y_preds[key], False)
        best_threshold_list.append(best_threshold)

        # Compute binary predictions from probability predictions with best threshold
        y_preds[key] = y_preds[key] > best_threshold

        # Compute metrics that require binary predictions
        mcc = matthews_corrcoef(y_trues[key], y_preds[key])
        mcc_list.append(mcc)
        precision = precision_score(y_trues[key], y_preds[key], zero_division=0)
        precision_list.append(precision)
        recall = recall_score(y_trues[key], y_preds[key])
        recall_list.append(recall)

    top1_mean = average(top1_list)
    top2_mean = average(top2_list)
    roc_auc_mean = average(roc_auc_list)
    best_threshold_mean = average(best_threshold_list)
    mcc_mean = average(mcc_list)
    precision_mean = average(precision_list)
    recall_mean = average(recall_list)

    data = [
        hdim,
        dropout,
        lr,
        wd,
        bs,
        round(best_threshold_mean, 3),
        round(mcc_mean, 3),
        round(top1_mean, 3),
        round(top2_mean, 3),
        round(precision_mean, 3),
        round(recall_mean, 3),
        round(roc_auc_mean, 3),
    ]

    if os.path.isfile(os.path.join(output_directory, results_file_name)) == False:
        with open(
            os.path.join(output_directory, results_file_name),
            "w",
            encoding="UTF8",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Dimension of Hidden Layers",
                    "Dropout",
                    "Learning Rate",
                    "Weight Decay",
                    "Batch Size",
                    "Optimal Threshold",
                    "MCC",
                    "Top1 Accuracy",
                    "Top2 Accuracy",
                    "Precision",
                    "Recall",
                    "ROC AUC Score",
                ]
            )
    with open(
        os.path.join(output_directory, results_file_name),
        "a",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(data)


def save_predict(
    outdir,
    y_preds,
    y_trues,
    opt_thresholds,
):

    results = {}
    
    y_preds_bin = {}
    for threshold in opt_thresholds:
        for key in y_preds:
            for y_pred in y_preds[key]:
                y_preds_bin.setdefault(key,[]).append(int(y_pred > threshold))

    y_preds_voted = {}
    for key in y_preds_bin:
        y_preds_voted[key] = Counter(y_preds_bin[key]).most_common()[0][0]


    mcc = matthews_corrcoef(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values()))
    precision = precision_score(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values()))
    recall = recall_score(y_true=list(y_trues.values()), y_pred=list(y_preds_voted.values()))

    results["MCC"] = mcc
    results["Precision"] = precision
    results["Recall"] = recall

    with open(
        os.path.join(outdir, "results.txt"),
        "w",
        encoding="UTF8",
    ) as f:
        f.write(json.dumps(results))

    # Save molecular identifiers, atom identifiers, true labels, and predicted labels of single atoms to csv file
    # (serves results visualization purposes)
    rows = zip(
        [k for (k,_) in y_trues.keys()], 
        [k for (_,k) in y_trues.keys()], 
        list(y_trues.values()), 
        list(y_preds_voted.values())
    )
    with open(
        os.path.join(outdir, "predictions.csv"),
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "mol_id",
                "atom_id",
                "true_label",
                "predicted_label",
            )
        )
        writer.writerows(rows)
    f.close()