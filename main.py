import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, \
    auc, roc_curve, RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
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
    train_dataset, test_dataset = train_test_split(dataset, test_size=(1/10), random_state=42, shuffle=True)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=(1/9), random_state=42, shuffle=True)

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

    gin_model = GIN(in_dim=dataset.num_features, h_dim=128, out_dim=dataset.num_classes).to(device)
    print(gin_model)

    for epoch in range(30):
        training_loss = train(gin_model, train_loader, class_weights, lr=1e-3, weight_decay=1e-4)
        val_pred, val_true = test(gin_model, val_loader)
        val_mcc = matthews_corrcoef(val_true, val_pred)
        print(f'Epoch: {epoch}, Training Loss: {training_loss}, Validation MCC: {val_mcc}.')

    # Compute and display ROC curve
    fpr, tpr, _ = roc_curve(val_true, val_pred)
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
    plt.savefig('output/roc_curve.png')

    # Compute and display precision/recall curve
    PrecisionRecallDisplay.from_predictions(val_true, val_pred)
    plt.savefig('output/precision_recall_curve.png')

    # Compute and display confusion matrix
    ConfusionMatrixDisplay.from_predictions(val_true, val_pred)
    plt.savefig('output/confusion_matrix.png')

if __name__ == "__main__":
    main()
