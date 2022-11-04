import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, \
    jaccard_score, precision_score, recall_score, roc_auc_score, \
        classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily

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
    #process_data(path='data/dataset_new.sdf')

    # Create/Load Custom PyTorch Geometric Dataset
    dataset = SOM(root='data')

    # Print dataset info
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of node features: {dataset.num_node_features}')
    print(f'Number of edge features: {dataset.num_edge_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Compute homophily
    loader = DataLoader(dataset, batch_size=len(dataset))
    for data in loader: print(f'Homophily: {homophily(data.edge_index, data.y)}')

    # Training/Test Split
    train_val_dataset, test_dataset = train_test_split(dataset, test_size=(1/10), random_state=42, shuffle=True)
    print(f'Test set: {len(test_dataset)} molecules.')

    # Initialize cross-validation metrics
    k = 1  # number of cross-validation runs
    if k == 1: test_size = 1/9
    else: test_size = 1/k
    mcc = []
    accuracy = []
    jaccard = []
    precision = []
    recall = []
    roc_auc = []

    # Set parameters
    h_dim = 16
    epochs = 300
    lr = 1e-4
    weight_decay = 1e-3

    for _ in range(k):

        # Initialize model
        model = GIN(in_dim=dataset.num_features, h_dim=h_dim, edge_dim=dataset.num_edge_features).to(device)
        #model = GAT(in_dim=dataset.num_features, h_dim=16, num_heads=4, edge_dim=dataset.num_edge_features).to(device)

        # Training/Validation Split
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=test_size, random_state=42, shuffle=True)
        print(f'Training set: {len(train_dataset)} molecules.')
        print(f'Validation set: {len(val_dataset)} molecules.')

        #  Data Loader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # Compute class weights of the training set:
        class_weights = 0
        total_num_instances = 0
        for data in train_loader:
            class_weights += compute_class_weight(class_weight='balanced', classes=np.unique(data.y), y=np.array(data.y)) * len(data.y)
            total_num_instances += len(data.y)
        class_weights /= total_num_instances

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, delta=0.005)

        # Train and validate model
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            train_loss = train(model, train_loader, lr=lr, weight_decay=weight_decay, device=device)
            train_losses.append(train_loss.item())
            val_loss, val_pred, val_true = test(model, val_loader, device=device, threshold=0.5)
            val_losses.append(val_loss.item())
            val_mcc = matthews_corrcoef(val_true, val_pred)
            val_acc = accuracy_score(val_true, val_pred)
            val_jacc = jaccard_score(val_true, val_pred)
            val_prec = precision_score(val_true, val_pred, zero_division=0)
            val_rec = recall_score(val_true, val_pred)
            val_roc_auc = roc_auc_score(val_true, val_pred)
            print(  f'Epoch: {epoch}, '
                    f'Train Loss: {train_loss:.3f}, '
                    f'Val Loss: {val_loss:.3f}, '
                    f'MCC: {val_mcc:.2f}, '
                    f'Top-1-Accuracy : {val_acc:.2f}, '
                    f'Jaccard Score: {val_jacc:.2f}, '
                    f'Precision {val_prec:.2f}, '
                    f'Recall: {val_rec:.2f}, '
                    f'AUROC {val_roc_auc:.2f}.')

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

    # Compute and print averaged metrics
    mcc_avg = np.average(np.array(mcc))
    accuracy_avg = np.average(np.array(accuracy))
    jaccard_avg = np.average(np.array(jaccard))
    precision_avg = np.average(np.array(precision))
    recall_avg = np.average(np.array(recall))
    roc_auc_avg = np.average(np.array(roc_auc))

    with open("results.txt", "w") as f:
        f.write(f'Cross-validation with k={k}'
                f'MCC: {mcc_avg}\n'
                f'Accuracy: {accuracy_avg}\n'
                f'Jaccard: {jaccard_avg}\n'
                f'Precision: {precision_avg}\n'
                f'Recall: {recall_avg}\n'
                f'ROC-AUC-Score: {roc_auc_avg}')

    # Plot training and validtion losses
    plt.plot(np.arange(0, epochs, 1), train_losses, linestyle='-', linewidth=1, color ='orange', label='Training Loss')
    plt.plot(np.arange(0, epochs, 1), val_losses, linestyle='-', linewidth=1, color ='blue', label='Validation Loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss.png')

    # Compute and plot ROC curve
    roc_curve_display(val_true, val_pred, 'output/roc_curve.png')

    # Compute and plot precision/recall curve
    PrecisionRecallDisplay.from_predictions(val_true, val_pred)
    plt.savefig('output/precision_recall_curve.png')

    # Compute and plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(val_true, val_pred)
    plt.savefig('output/confusion_matrix.png')

if __name__ == "__main__":
    main()
