aweSOM
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/som_gnn/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/som_gnn/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/SOM_GNN/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/SOM_GNN/branch/main)


A Graph Neural Network (GNN) model for the prediction of Sites of Metabolism (SoMs) in small organic molecules.

### Installation

1. Clone the repository and cd into the repository root:

```git clone https://github.com/molinfo-vienna/aweSOM.git```

```cd aweSOM```

2. Create a conda environment with the required dependencies:

```conda env create -f awesom-env.yml```

3. Activate the environment:

```conda activate awesom-env```

4. Install awesom package with ```pip install -e .```

### Usage

#### Dataset Generation

To load new data for model training/testing and/or prediction purposes run the following before running ```train``` or ```predict```:

```python scripts/preprocess.py -i INPUT_DIRECTORY -o OUTPUT_DIRECTORY -w NUMBER_WORKERS -k KIT -tl TRUE_LABELS -v VERBOSE```

```INPUT_DIRECTORY``` is the directory of the unprocessed data. Only ```.sdf``` and ```.smiles``` input files are currently supported.

```OUTPUT_DIRECTORY``` is the directory where the preprocessed data will stored.

```WORKERS``` is the desired number of parallel workers. Please do not set this number higher than the number of molecules in the data file.

```KIT``` is the desired chemistry tool kit for preprocessing the data. Choose between ```RDKit``` and ```CDPKit```.

```TRUE_LABELS``` (True/False) indicates whether the input data has true labels or not. Set to ```True``` when preprocessing training data and ```False``` when preprocessing data for making predictions.

```VERBOSE``` should be set to the desired verbosity level, e.g. ```INFO```.

```python scripts/preprocess.py -i data/raw/zaretzki.sdf -o data/preprocessed/CDPKit/zaretzki -w 12 -k CDPKit -tl True -v INFO```
```python scripts/preprocess.py -i data/raw/propanolol.smiles -o data/preprocessed/RDKit/propanolol -w 1 -k RDKit -tl False -v INFO```

#### Model Training

Running the following will take preprocessed training data and run internal/external cross validation to determine the optimal hyperparameters with Optuna. The best model per external cross-validation fold will be saved in the corresponding subdirectory (model.pt), along with a text file (info.txt) holding the best hyperparameters and a plot of the validation and training losses during the final training phase (loss.png). Trained models can then be used to make predictions on preprocessed data.

```python scripts/train.py -i INPUT_DIRECTORY -o OUTPUT_DIRECTORY -m MODEL_TYPE -l LOSS_FUNCTION -b BATCH_SIZE -e EPOCHS -nt NUMBER_TRIALS -nif NUMBER_INTERNAL_FOLDS -nef NUMBER_EXTERNAL_FOLDS -v VERBOSE```

```INPUT_DIRECTORY``` is the directory where the preprocessed training data is stored.

```OUTPUT_DIRECTORY``` is the directory where the trained models and their metadata will be stored.

Supported ```MODEL_TYPE``` include ```GIN```, ```GINE```, ```GATv2```, ```MF``` and ```TF```.

GIN refers to PyTorch Geometric's GINConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv

GINE refers to PyTorch Geometric's GINEConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html#torch_geometric.nn.conv.GINEConv

GATv2 refers to PyTorch Geometric's GATv2Conv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv

MF refers to PyTorch Geometric's MFConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv

TF refers to PyTorch Geometric's TransformerConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html#torch_geometric.nn.conv.TransformerConv

Supported ```LOSS_FUNCTION``` inlcude ```BCE```, ```weighted_BCE```, ```MCC_BCE``` and ```focal```.

```NUMBER_TRIALS``` is the desired number of Optuna trials. Recommended setting: 100.

```NUMBER_INTERNAL_FOLDS``` is the desired number of folds for the internal cross-validation. Recommended setting: 5.

```NUMBER_EXTERNAL_FOLDS``` is the desired number of folds for the external cross-validation. Recommended setting: 5.

```VERBOSE``` should be set to the desired verbosity level, e.g. ```INFO```.

Example:

```python scripts/train.py -i data_preprocessed/RDKit/zaretzki -o models/RDKit/zaretzki/gine/bce -m GINE -l BCE -b 64 -e 500 -nt 100 -nif 5 -nef 5 -v INFO```

#### Predicting SoMs

To predict the Sites of Metabolism of one or multiple molecules run:

```python scripts/predict.py -i INPUT_DIRECTORY o OUTPUT_DIRECTORY -m MODELS_DIRECTORY -v VERBOSE```

Example:

```python scripts/predict.py -i data_preprocessed/RDKit/capsaicin -o output/RDKit/capsaicin/zaretzki/gine/bce -m models/RDKit/zaretzki/gine/bce -v INFO```

### License

This project is licensed under MIT license.
