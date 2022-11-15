import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, \
    jaccard_score, precision_score, recall_score, roc_auc_score, \
        ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily
from tqdm import tqdm

from src.process_input_data import process_data
from src.pyg_dataset_creator import SOM
from src.graph_neural_nets import GIN, GAT, train_oversampling, train, test
from src.utils import plot_losses, plot_roc_curve


def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = 'data/'
    data_name = 'dataset_preprocessed.sdf'
    output_path = 'output/'

    # Process SDF input data to create PyTorch Geometric custom dataset
    #process_data(data_path=data_path, data_name=data_name, output_path=output_path)

    # Create/Load Custom PyTorch Geometric Dataset
    dataset = SOM(root=data_path)

    # Print dataset info
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of node features: {dataset.num_node_features}')
    print(f'Number of edge features: {dataset.num_edge_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Compute homophily
    loader = DataLoader(dataset, batch_size=len(dataset))
    for data in loader: print(f'Homophily: {homophily(data.edge_index, data.y):.2f}')

    # Training/Validation/Test Split
    #train_val_dataset, test_dataset = train_test_split(dataset, test_size=1/10, random_state=42, shuffle=True)
    #train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=1/10, random_state=42, shuffle=True)
    train_dataset, val_dataset = train_test_split(dataset, test_size=1/10, random_state=42, shuffle=True)
    print(f'Training set: {len(train_dataset)} molecules.')
    print(f'Validation set: {len(val_dataset)} molecules.')
    #print(f'Test set: {len(test_dataset)} molecules.')

    # Set parameters
    h_dim = 16
    epochs = 400
    lr = 1e-4
    weight_decay = 1e-4

    # Initialize model
    model = GIN(in_dim=dataset.num_features, h_dim=h_dim, edge_dim=dataset.num_edge_features).to(device)
    #model = GAT(in_dim=dataset.num_features, h_dim=16, num_heads=4, edge_dim=dataset.num_edge_features).to(device)

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


    """ ---------- Train Model ---------- """

    dir = output_path+'dicebce/'
    train_losses = []
    val_losses = []
    print('Training...')
    for _ in tqdm(range(epochs)):
        train_loss = train(model, train_loader, lr, weight_decay, device)
        train_losses.append(train_loss.item())
        val_loss, _, _ ,_ = test(model, val_loader, device)
        val_losses.append(val_loss.item())
    print('Training done!')
    torch.save(model.state_dict(), output_path+'model.pt')

    # Plot training and validation losses
    plot_losses(train_losses, val_losses, path=dir+'loss.png')


    """ ---------- Evaluate Model ---------- """

    model.load_state_dict(torch.load(output_path+'model.pt'))
    val_loss, val_pred, val_mol_ids, val_true = test(model, val_loader, device)

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

    # Compute and plot precision/recall curve
    PrecisionRecallDisplay.from_predictions(val_true, val_pred)
    plt.savefig(dir+'precision_recall_curve.png')

    # Compute and plot ROC-AUC score and ROC-curve, get best threshold
    val_roc_auc = roc_auc_score(val_true, val_pred)
    best_threshold = plot_roc_curve(val_true, val_pred, dir+'roc_curve.png')

    # Compute binary predictions from probability predictions with best threshold
    val_pred = ((val_pred > best_threshold)[:,0])

    # Compute metrics tht require binry predictions
    val_mcc = matthews_corrcoef(val_true, val_pred)
    val_jacc = jaccard_score(val_true, val_pred)
    val_prec = precision_score(val_true, val_pred, zero_division=0)
    val_rec = recall_score(val_true, val_pred)

    with open(dir+'results.txt', 'w') as f:
        f.write(f'Dimension of hidden layer: {h_dim}\n'
                f'Number of training epochs: {epochs}\n'
                f'Learning rate: {lr}\n'
                f'Weight decay: {weight_decay}\n'
                f'Best threshold: {best_threshold}\n'
                f'Validation MCC: {val_mcc:.2f}\n'
                f'Validation Top1-Accuracy: {val_acc1:.2f}\n'
                f'Validation Top2-Accuracy: {val_acc2:.2f}\n'
                f'Validation Jaccard Score: {val_jacc:.2f}\n'
                f'Validation Precision: {val_prec:.2f}\n'
                f'Validation Recall: {val_rec:.2f}\n'
                f'Validation ROC-AUC-Score: {val_roc_auc:.2f}')

    # Compute and plot confusion matrix 
    ConfusionMatrixDisplay.from_predictions(val_true, val_pred)
    plt.savefig(dir+'confusion_matrix.png')

if __name__ == "__main__":
    main()
