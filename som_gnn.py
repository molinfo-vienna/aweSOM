import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, \
    jaccard_score, precision_score, recall_score, roc_auc_score, \
        ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch_geometric.loader import DataLoader

from src.process_input_data import process_data
from src.pyg_dataset_creator import SOM
from src.graph_neural_nets import GIN, GAT, train, test
from src.utils import EarlyStopping, roc_curve_display


def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process SDF input data to create PyTorch Geometric custom dataset
    #process_data(path='data/dataset.sdf')

    # Create/Load Custom PyTorch Geometric Dataset
    dataset = SOM(root='data')

    # Print dataset info
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Set out test set
    train_dataset, test_dataset = train_test_split(dataset, test_size=(1/10), random_state=42, shuffle=True)
    print(f'Test set: {len(test_dataset)} molecules.')

    # Initialize cross-validation metrics
    k = 3  # number of cross-validation runs
    mcc = []
    accuracy = []
    jaccard = []
    precision = []
    recall = []
    roc_auc = []

    for _ in range(k):

        # Initialize model
        model = GIN(in_dim=dataset.num_features, h_dim=16, out_dim=dataset.num_classes).to(device)
        #model = GAT(in_dim=dataset.num_features, h_dim=64, out_dim=dataset.num_classes, num_heads=4).to(device)

        # Training/Validation/Test Split
        train_dataset, val_dataset = train_test_split(dataset, test_size=(1/k), random_state=42, shuffle=True)
        print(f'Training set: {len(train_dataset)} molecules.')
        print(f'Validation set: {len(val_dataset)} molecules.')

        #  Data Loader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # Compute class weights of the validation set (used for computing validation loss):
        class_weights = 0
        total_num_instances = 0
        for data in val_loader:
            class_weights += compute_class_weight(class_weight='balanced', classes=np.unique(data.y), y=np.array(data.y)) * len(data.y)
            total_num_instances += len(data.y)
        class_weights /= total_num_instances

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, delta=0.005)

        # Train and validate model
        for epoch in range(50):
            train_loss = train(model, train_loader, class_weights, lr=1e-4, weight_decay=1e-4, device=device)
            val_loss, val_pred, val_true = test(model, val_loader, class_weights, device=device)

            val_mcc = matthews_corrcoef(val_true, val_pred)
            val_acc = accuracy_score(val_true, val_pred)
            val_jacc = jaccard_score(val_true, val_pred)
            val_prec = precision_score(val_true, val_pred)
            val_rec = recall_score(val_true, val_pred)
            val_roc_auc = roc_auc_score(val_true, val_pred)
            print(  f'Epoch: {epoch}, '
                    f'Train Loss: {train_loss:.3f}, '
                    f'Val Loss: {val_loss:.3f}, '
                    f'Val MCC: {val_mcc:.2f}, '
                    f'Val Top-1-Accuracy : {val_acc:.2f}, '
                    f'Val Jaccard Score: {val_jacc:.2f}, '
                    f'Val Precision {val_prec:.2f}, '
                    f'Val Recall: {val_rec:.2f}, '
                    f'Val AUROC {val_roc_auc:.2f}.')

            #early_stopping(criterion=val_loss, opt_mode='min')
            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    break

        mcc.append(val_mcc)
        accuracy.append(val_acc)
        jaccard.append(val_jacc)
        precision.append(val_prec)
        recall.append(val_rec)
        roc_auc.append(val_roc_auc)

    # Compute averaged metrics
    mcc_avg = np.average(np.array(mcc))
    accuracy_avg = np.average(np.array(accuracy))
    jaccard_avg = np.average(np.array(jaccard))
    precision_avg = np.average(np.array(precision))
    recall_avg = np.average(np.array(recall))
    roc_auc_avg = np.average(np.array(roc_auc))

    print(  f'MCC: {mcc_avg}\n'
            f'Accuracy: {accuracy_avg}\n'
            f'Jaccard: {jaccard_avg}\n'
            f'Precision: {precision_avg}\n'
            f'Recall: {recall_avg}\n'
            f'ROC-AUC-Score: {roc_auc_avg}')

    # Compute and display ROC curve
    roc_curve_display(val_true, val_pred, 'output/roc_curve.png')

    # Compute and display precision/recall curve
    PrecisionRecallDisplay.from_predictions(val_true, val_pred)
    plt.savefig('output/precision_recall_curve.png')

    # Compute and display confusion matrix
    ConfusionMatrixDisplay.from_predictions(val_true, val_pred)
    plt.savefig('output/confusion_matrix.png')

if __name__ == "__main__":
    main()
