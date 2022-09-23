from load_data.process_input_data import process_data
from load_data.pyg_dataset_creator import SOM
from torch_geometric.loader import DataLoader


def main():

    """
        -------------------------------------
        ------------Loading Data-------------
        ------------------------------------- 
    """

    # Process SDF input data to create PyTorch Geometric custom dataset
    process_data(path='data/db_preprocessed.sdf')

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


if __name__ == "__main__":
    main()