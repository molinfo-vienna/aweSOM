import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from collections import Counter
from sklearn.metrics import (
    auc,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statistics import mean

#################### Neural network related utility functions ####################
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't 
    improve after a given patience."""
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): How many epcohs to wait after last time 
                            validation loss improved before stopping.
                            Default: 5
            delta (float):  Minimum change in the monitored quantity 
                            to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each 
                            validation loss improvement. 
                            Default: False       
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


#################### Loss related utility functions ####################

class weighted_BCE_Loss(torch.nn.Module):
    def __init__(self):
        super(weighted_BCE_Loss, self).__init__()

    def forward(self, output, target, class_weights=None):
        if class_weights is not None:
            assert len(class_weights) == 2
            
            loss = class_weights[1] * (target * torch.log(output)) + \
                class_weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))


class MCC_BCE_Loss(torch.nn.Module):
    def __init__(self):
        super(MCC_BCE_Loss, self).__init__()

    def forward(self, predictions, targets):
        tp = torch.sum(predictions * targets)
        tn = torch.sum((1 - predictions) * (1 - targets))
        fp = torch.sum((1 - targets) * predictions)
        fn = torch.sum(targets * (1 - predictions))
        MCC = (tp * tn - fp * fn) / (torch.sqrt(tp + fp) * torch.sqrt(tp + fn) * torch.sqrt(tn + fp) * torch.sqrt(tn + fn))
        
        MCC_loss = 1 - MCC
        BCE_loss = F.binary_cross_entropy(predictions, targets, reduction="sum")
        return MCC_loss + BCE_loss
    
    
class FocalLoss(torch.nn.Module):
    """
    Implementation of FocalLoss for imbalanced datasets with binary labels.
    """

    def __init__(self, alpha:float=0.9, gamma:float=2.0, reduction:str='mean',):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Function that computes Binary Focal loss.
        .. math::
            \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
        where:
        - :math:p_t is the model's estimated probability for each class.
        Args:
            input: input data tensor of arbitrary shape.
            target: the target tensor with shape matching input.
            alpha: Weighting factor for the rare class :math:\alpha \in [0, 1].
            gamma: Focusing parameter :math:\gamma >= 0.
            reduction: Specifies the reduction to apply to the
            output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the sum of the output will be divided by
            the number of elements in the output, ``'sum'``: the output will be
            summed.
            eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.
            pos_weight: a weight of positive examples.
            It’s possible to trade off recall and precision by adding weights to positive examples.
            Must be a vector with length equal to the number of classes.
        Returns:
            the computed loss.
        Examples:
            >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
            >>> logits = torch.tensor([[[6.325]],[[5.26]],[[87.49]]])
            >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
            >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
            tensor(21.8725)
        """

        probs_pos = input.sigmoid()
        probs_neg = (-input).sigmoid()

        log_probs_pos = torch.nn.functional.logsigmoid(input)
        log_probs_neg = torch.nn.functional.logsigmoid(-input)

        loss_tmp = (
            -self.alpha * probs_neg.pow(self.gamma) * target * log_probs_pos
            - (1 - self.alpha) * probs_pos.pow(self.gamma) * (1.0 - target) * log_probs_neg
        )

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss


#################### General utility functions ####################

def make_dir(dir):
    for folder in ["train", "test"]:
        os.makedirs(os.path.join(dir, "preprocessed", folder))

def average(lst):
    return sum(lst) / len(lst)

def seed_everything(seed=42):
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
    final_num_epochs,
    val_loss,
    y_pred,
    y_true,
    mol_id,
    atom_id,
    hyperparams,
    hyperparams_var_name,
):

    # # Compute top1 accuracy
    # pred_top1 = []
    # for id in np.unique(mol_id):
    #     mask = id == mol_id
    #     idx = np.argpartition(y_pred[mask], -1)[-1:]
    #     if y_true[mask][idx[0]]:
    #         pred_top1.append(1)
    #     else:
    #         pred_top1.append(0)
    # top1 = np.sum(pred_top1) / len(np.unique(mol_id))

    # # Compute top2 accuracy
    # pred_top2 = []
    # for id in np.unique(mol_id):
    #     mask = id == mol_id
    #     idx = np.argpartition(y_pred[mask], -2)[-2:]
    #     if y_true[mask][idx[0]] or y_true[mask][idx[1]]:
    #         pred_top2.append(1)
    #     else:
    #         pred_top2.append(0)
    # top2 = np.sum(pred_top2) / len(np.unique(mol_id))

    # Compute PR-AUC
    pr_auc = average_precision_score(y_true, y_pred)

    # Compute ROC-AUC, get best threshold
    roc_auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    # Compute binary predictions from probability predictions with best threshold
    y_pred_bin = y_pred > best_threshold

    # Compute metrics that require binary predictions
    mcc = matthews_corrcoef(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin, zero_division=0)
    recall = recall_score(y_true, y_pred_bin)

    # Save molecular identifiers (mol_id), true labels (y_true), and predicted labels (y_pred) to a csv file:
    rows = zip(mol_id, atom_id, y_true, y_pred, y_pred_bin)
    with open(
        os.path.join(output_subdirectory, "predictions.csv"),
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

    data = [str(output_subdirectory)]
    data.extend([hp for hp in hyperparams])
    data.extend([
                fold_num,
                final_num_epochs,
                round(val_loss, 3),
                round(best_threshold, 3),
                round(mcc, 3),
                round(pr_auc, 3),
                # round(top1, 3),
                # round(top2, 3),
                round(precision, 3),
                round(recall, 3),
                round(roc_auc, 3),
                ]
               )

    header = ["ResultsFolder"]
    header.extend(hyperparams_var_name)
    header.extend([
                    "Fold",
                    "Epochs",
                    "ValLoss",
                    "OptThreshold",
                    "MCC",
                    "PRAUC",
                    # "Top1",
                    # "Top2",
                    "Precision",
                    "Recall",
                    "ROCAUC",
                ])
    
    if os.path.isfile(os.path.join(output_directory, results_file_name)) == False:
        with open(
            os.path.join(output_directory, results_file_name),
            "w",
            encoding="UTF8",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(header)
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
    y_preds,
    y_trues,
    mol_ids,
    hyperparams,
    hyperparams_var_name,
):

    # top1_list = []
    # top2_list = []
    pr_auc_list = []
    roc_auc_list = []
    best_threshold_list = []
    mcc_list = []
    precision_list = []
    recall_list = []

    for key in y_preds:

        # # Compute top1 accuracy
        # pred_top1 = []
        # for id in np.unique(mol_ids[key]):
        #     mask = id == mol_ids[key]
        #     idx = np.argpartition(y_preds[key][mask], -1)[-1:]
        #     if y_trues[key][mask][idx[0]]:
        #         pred_top1.append(1)
        #     else:
        #         pred_top1.append(0)
        # top1 = np.sum(pred_top1) / len(np.unique(mol_ids[key]))
        # top1_list.append(top1)

        # # Compute top2 accuracy
        # pred_top2 = []
        # for id in np.unique(mol_ids[key]):
        #     mask = id == mol_ids[key]
        #     idx = np.argpartition(y_preds[key][mask], -2)[-2:]
        #     if y_trues[key][mask][idx[0]] or y_trues[key][mask][idx[1]]:
        #         pred_top2.append(1)
        #     else:
        #         pred_top2.append(0)
        # top2 = np.sum(pred_top2) / len(np.unique(mol_ids[key]))
        # top2_list.append(top2)

        # Compute PR-AUC
        pr_auc = average_precision_score(y_trues[key], y_preds[key])
        pr_auc_list.append(pr_auc)

        # Compute ROC-AUC and get best threshold via ROC curve
        roc_auc = roc_auc_score(y_trues[key], y_preds[key])
        roc_auc_list.append(roc_auc)
        fpr, tpr, thresholds = roc_curve(y_trues[key], y_preds[key])
        best_threshold = thresholds[np.argmax(tpr - fpr)]
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

    # top1_mean = average(top1_list)
    # top2_mean = average(top2_list)
    pr_auc_mean = average(pr_auc_list)
    roc_auc_mean = average(roc_auc_list)
    best_threshold_mean = average(best_threshold_list)
    mcc_mean = average(mcc_list)
    precision_mean = average(precision_list)
    recall_mean = average(recall_list)

    data = [hp for hp in hyperparams]
    data.extend([
                    round(best_threshold_mean, 3),
                    round(mcc_mean, 3),
                    round(pr_auc_mean, 3),
                    # round(top1_mean, 3),
                    # round(top2_mean, 3),
                    round(precision_mean, 3),
                    round(recall_mean, 3),
                    round(roc_auc_mean, 3),
                ])
    
    header = hyperparams_var_name
    header.extend([
                    "OptThreshold",
                    "MCC",
                    "PRAUC",
                    # "Top1",
                    # "Top2",
                    "Precision",
                    "Recall",
                    "ROCAUC",
                ])

    if os.path.isfile(os.path.join(output_directory, results_file_name)) == False:
        with open(
            os.path.join(output_directory, results_file_name),
            "w",
            encoding="UTF8",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(header)
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
    # Compute binary labels (y_preds_bin) from SoM probabilities (y_preds)
    y_preds_bin = {key: [int(y_pred > threshold) for y_pred in preds] for key, preds in y_preds.items() for threshold in opt_thresholds}

    # Let the models in the ensemble classifier vote for the final binary label
    y_preds_voted = {key: Counter(preds).most_common(1)[0][0] for key, preds in y_preds_bin.items()}

    # Average SoM probability outputs
    y_preds_avg = [mean(preds) for preds in y_preds.values()]

    # Average optimal thresholds
    opt_thresholds_avg = np.round(np.average(opt_thresholds), 2)
    # Compute standard deviation in optimal thresholds
    opt_thresholds_std = np.round(np.std(opt_thresholds), 2)

    # Compute top1 and top2 accuracies
    # mol_ids = np.unique([a for a,b in y_trues.keys()])
    # pred_top1 = []
    # pred_top2 = []
    # for mol_id in mol_ids: 
    #     mask = [a == mol_id for a,b in y_trues.keys()]
    #     idx = np.argpartition([y_preds_avg[i] for i, x in enumerate(mask) if x], -1)[-1:]
    #     if [list(y_trues.values())[i] for i, x in enumerate(mask) if x][idx[0]]:
    #         pred_top1.append(1)
    #     else:
    #         pred_top1.append(0)
    #     idx = np.argpartition([y_preds_avg[i] for i, x in enumerate(mask) if x], -2)[-2:]
    #     if ([list(y_trues.values())[i] for i, x in enumerate(mask) if x][idx[0]] or 
    #         [list(y_trues.values())[i] for i, x in enumerate(mask) if x][idx[1]]):
    #         pred_top2.append(1)
    #     else:
    #         pred_top2.append(0)
    # top1 = np.sum(pred_top1) / len(mol_ids)
    # top2 = np.sum(pred_top2) / len(mol_ids)

    results = {}
    y_true=list(y_trues.values())
    y_pred=list(y_preds_voted.values())
    results["mcc"] = round(matthews_corrcoef(y_true, y_pred),2)
    results["precision"] = round(precision_score(y_true, y_pred),2)
    results["recall"] = round(recall_score(y_true, y_pred),2)
    # results["top1"] = round(top1,2)
    # results["top2"] = round(top2,2)
    results["optThresholdAvg"] = round(opt_thresholds_avg,2)
    results["optThresholdStdev"] = round(opt_thresholds_std,2)

    with open(
        os.path.join(outdir, "results.txt"),
        "w",
        encoding="UTF8",
    ) as f:
        f.write(json.dumps(results))

    # Save molecular identifiers, atom identifiers, true labels, 
    # predicted binary labels and (averaged) predicted som-probabilities of 
    # single atoms to csv file
    rows = zip(
        [k for (k,_) in y_trues.keys()], 
        [k for (_,k) in y_trues.keys()], 
        list(y_trues.values()), 
        list(y_preds_voted.values()),
        y_preds_avg,
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
                "predicted_binary_label",
                "averaged_som_probability",
            )
        )
        writer.writerows(rows)


def save_test(
    mccs, 
    precisions, 
    recalls, 
    top1s, 
    top2s, 
    out, 
):

    rows = zip(
        mccs, 
        precisions, 
        recalls, 
        top1s, 
        top2s, 
    )
    with open(
        os.path.join(out, "individual_results.csv"),
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            (
            "mcc", 
            "precision", 
            "recall", 
            "top1", 
            "top2", 
            )
        )
        writer.writerows(rows)

    row_metrics = [
        np.round(np.average(mccs), 2), 
        np.round(np.average(precisions), 2),
        np.round(np.average(recalls), 2),
        np.round(np.average(top1s), 2),
        np.round(np.average(top2s), 2),
    ]
    row_std = [
        np.round(np.std(mccs), 2),
        np.round(np.std(precisions), 2),
        np.round(np.std(recalls), 2),
        np.round(np.std(top1s), 2),
        np.round(np.std(top2s), 2),
    ]
    metric_names = [
        "mcc", 
        "precision", 
        "recall", 
        "top1", 
        "top2", 
        ]
    with open(
        os.path.join(out, "results_summary.csv"),
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            (
            "metric", 
            "avg", 
            "stdev", 
            )
        )
        writer.writerows(zip(metric_names, row_metrics, row_std))
