import os
from sklearn.utils.class_weight import compute_class_weight
import time
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.graph_neural_nets import GIN, GAT, train, train_oversampling, test
from src.utils import EarlyStopping, plot_losses, save_individual_results, save_average_results


def hp_opt(device, dataset, train_data, val_data, output_directory, results_file_name, data_name, model_name, \
            h_dim, dropout, num_heads, neg_slope, epochs, lr, wd, batch_size, oversampling, patience, delta):

    timestamp = int(time.time())
    output_subdirectory = os.path.join(output_directory, str(timestamp))
    os.mkdir(os.path.join(os.getcwd(), output_subdirectory))
        
    # Initialize model
    if model_name == "GIN":
        model = GIN(in_dim=dataset.num_features, h_dim=h_dim, edge_dim=dataset.num_edge_features, dropout=dropout).to(device)
    if model_name == "GAT":
        model = GAT(in_dim=dataset.num_features, h_dim=h_dim, edge_dim=dataset.num_edge_features, num_heads=num_heads, neg_slope=neg_slope, dropout=dropout).to(device)

    #  Training and Validation Data Loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Compute class weights of the training set:
    # class_weights = 0
    # total_num_instances = 0
    # for data in train_loader:
    #     class_weights += compute_class_weight(class_weight='balanced', classes=np.unique(data.y), y=np.array(data.y)) * len(data.y)
    #     total_num_instances += len(data.y)
    # class_weights /= total_num_instances

    """ ---------- Train Model ---------- """

    early_stopping = EarlyStopping(patience, delta)

    train_losses = []
    val_losses = []
    print('Training...')
    for epoch in tqdm(range(epochs)):
        if oversampling:
            train_loss = train_oversampling(model, train_loader, lr, wd, device)
        else:
            train_loss = train(model, train_loader, lr, wd, device)
        train_losses.append(train_loss.item())
        val_loss, _, _ ,_ = test(model, val_loader, device)
        val_losses.append(val_loss.item())
        early_stopping(val_loss)
        if early_stopping.early_stop:
            final_num_epochs = epoch
            break
    if early_stopping.early_stop == False:
        final_num_epochs = epochs
    torch.save(model.state_dict(), os.path.join(output_subdirectory, 'model.pt'))
    plot_losses(train_losses, val_losses, path=os.path.join(output_subdirectory, 'loss.png'))

    """ ---------- Validate Model ---------- """

    _, y_pred, mol_id, y_true = test(model, val_loader, device)

    save_individual_results(output_directory, output_subdirectory, results_file_name, timestamp, data_name, model_name, \
        h_dim, dropout, num_heads, neg_slope, final_num_epochs, lr, wd, batch_size, oversampling, y_pred[:,0], mol_id, y_true)


def testing(device, dataset, train_data, test_data, output_directory, data_name, model_name, \
            h_dim, dropout, num_heads, neg_slope, epochs, lr, wd, batch_size, oversampling, patience, delta):

    random_seeds = [123, 132, 213, 231, 312, 321]
    runs = []
    y_preds = {}
    mol_ids = {}
    y_trues = {}

    for rs in random_seeds:

        seed_everything(rs)

        timestamp = int(time.time())
        runs.append(timestamp)
        output_subdirectory = os.path.join(output_directory, str(timestamp))
        os.mkdir(os.path.join(os.getcwd(), output_subdirectory))
            
        # Initialize model
        if model_name == "GIN":
            model = GIN(in_dim=dataset.num_features, h_dim=h_dim, edge_dim=dataset.num_edge_features, dropout=dropout).to(device)
        if model_name == "GAT":
            model = GAT(in_dim=dataset.num_features, h_dim=h_dim, edge_dim=dataset.num_edge_features, num_heads=num_heads, neg_slope=neg_slope, dropout=dropout).to(device)

        #  Training and Test Data Loader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        # Compute class weights of the training set:
        # class_weights = 0
        # total_num_instances = 0
        # for data in train_loader:
        #     class_weights += compute_class_weight(class_weight='balanced', classes=np.unique(data.y), y=np.array(data.y)) * len(data.y)
        #     total_num_instances += len(data.y)
        # class_weights /= total_num_instances

        """ ---------- Train Model ---------- """

        early_stopping = EarlyStopping(patience, delta)

        train_losses = []
        test_losses = []
        print('Training...')
        for epoch in tqdm(range(epochs)):
            if oversampling:
                train_loss = train_oversampling(model, train_loader, lr, wd, device)
            else:
                train_loss = train(model, train_loader, lr, wd, device)
            train_losses.append(train_loss.item())
            test_loss, _, _ ,_ = test(model, test_loader, device)
            test_losses.append(test_loss.item())
            early_stopping(test_loss)
            if early_stopping.early_stop:
                final_num_epochs = epoch
                break
        if early_stopping.early_stop == False:
            final_num_epochs = epochs
        torch.save(model.state_dict(), os.path.join(output_subdirectory, 'model.pt'))
        plot_losses(train_losses, test_losses, path=os.path.join(output_subdirectory, 'loss.png'))

        """ ---------- Test Model ---------- """

        _, y_pred, mol_id, y_true = test(model, test_loader, device)

        y_preds[rs] = y_pred[:,0]
        mol_ids[rs] = mol_id
        y_trues[rs] = y_true

        save_individual_results(output_directory, output_subdirectory, "results_individual.csv", timestamp, data_name, model_name, \
            h_dim, dropout, num_heads, neg_slope, final_num_epochs, lr, wd, batch_size, oversampling, y_pred[:,0], mol_id, y_true)

    save_average_results(output_directory, "results_average.csv", runs, data_name, model_name, \
        h_dim, dropout, num_heads, neg_slope, lr, wd, batch_size, oversampling, y_preds, mol_ids, y_trues)