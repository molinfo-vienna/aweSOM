import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from load_data.process_input_data import process_data
from load_data.pyg_dataset_creator import SOM
from models.graph_neural_nets import GIN, train, test


def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    """
        --------------------------------------
        --------------Load Data---------------
        --------------------------------------
    """

    # Process SDF input data to create PyTorch Geometric custom dataset
    #process_data(path='data/db_preprocessed.sdf')

    # Create/Load Custom PyTorch Geometric Dataset
    dataset = SOM(root='data')

    # Print dataset info
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Training/Validation/Test Split
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset = dataset[int(len(dataset)*0.9):]

    print(f'Training set: {len(train_dataset)} graphs.')
    print(f'Validation set: {len(val_dataset)} graphs.')
    print(f'Test set: {len(test_dataset)} graphs.')

    #  Data Loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Compute class weights:
    class_weights = 0
    total_num_instances = 0
    for data in train_loader:
        class_weights += compute_class_weight(class_weight='balanced', classes=np.unique(data.y), y=np.array(data.y)) * len(data.y)
        total_num_instances += len(data.y)
    class_weights /= total_num_instances


    """
    --------------------------------------
      Initialize, Train, Test Neural Net 
    --------------------------------------
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    gin_model = GIN(in_dim=dataset.num_features, h_dim=32, out_dim=dataset.num_classes).to(device)
    print(gin_model)

    for epoch in range(20):
        training_loss = train(gin_model, train_loader, class_weights)
        val_pred, val_true = test(gin_model, val_loader)
        val_acc = accuracy_score(val_pred, val_true)
        print(f'Epoch: {epoch}, Training Loss: {training_loss}, Validation Accuracy: {val_acc}.')

    test_pred, test_true = test(gin_model, test_loader)
    test_acc = accuracy_score(test_pred, test_true)
    test_f1 = f1_score(test_pred, test_true)
    test_roc_auc = roc_auc_score(test_pred, test_true)
    print(  f'Classification accuracy on the test set: {test_acc}.\n'
            f'F1 score on the test set: {test_f1}.\n'
            f'ROC AUC score on the test set: {test_roc_auc}.\n')

    # Compute and display ROC curve
    fpr, tpr, _ = roc_curve(test_true, test_pred)
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
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

    # Compute and display confusion matrix
    cm = confusion_matrix(test_true, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()
