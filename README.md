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

### Usage

#### Dataset Generation

To load new data for model training/testing and/or prediction purposes run the following:

```python scripts/preprocess.py -f FILE.sdf -d DIRECTORY -s SPLIT -fc FEATURES_COMBINATION -v VERBOSE```

```FILE``` is the name of the data file. Only ```.sdf``` files are currently supported.

```DIRECTORY``` is the directory under which the unprocessed data is stored.

```SPLIT``` indicates the split ratio of training and test data. E.g. 20 means that 20% of the data will be stored as test data (recommended for training/testing purposes). We recommend setting ```SPLIT``` to 100 for predicting purposes, i.e. if you simply wish to apply the already trained model to your data.

```VERBOSE``` should be set to the desired verbosity level, e.g. ```INFO```.

```FEATURES_COMBINATION``` refers to the preferred featurization scheme, e.g., ```FC1```. A list of options is provided in ```fc.txt```.

The output of the preprocessing steps will be written in the ```DIRECTORY/preprocessed/train/``` and ```DIRECTORY/preprocessed/test/``` folders. These folders will be automatically created when executing ```preprocess.py```.

#### Model Training

To train a model with a specific set of hyperparameters run:

```python scripts/train_*.py -d DATA_DIRECTORY -lf LOSS_FUNCTION [HYPERPARAMETERS] -bs BATCH_SIZE -p PATIENCE -dt DELTA -o OUTPUT_DIRECTORY -hps HYPER_PARAMETER_SEARCH -v VERBOSE```

Be carefull that the set of hyperparameters can be different accross models (GIN, GAT, TransformerConv etc.). Please refer to the docs.

For example when training a GIN model with a specific set of hyperparameters DIMENSION_HIDDEN_LAYERS, DROPOUT, EPOCHS, LEARNING_RATE and WEIGHT_DECAY:

```python scripts/train_gin.py -d data -lf BCE -hd 64 -do 0.2 -e 1000 -lr 0.001 -wd 0.0001 -bs 64 -p 20 -dt 0 -o output/gin/bce -hps False -v VERBOSE```

Supported loss functions inlcude ```BCE```, ```weighted_BCE```, ```MCC_BCE``` and ```focal```.

Set ```-hps``` to ```True``` when performing hyperparameter search via shell script. When ```True```, the prompt asking whether to append, overwrite or cancel if the output folder already exists is deactivated and the results are automatically appended.


#### Model Testing

To test the performance of an ensemble classifier consisting of the *n* best trained models run:

```python scripts/test.py -d DATA_DIRECTORY -m MODEL -lf LOSS_FUNCTION -md MODELS_DIRECTORY -n NUMBER_MODELS -o OUTPUT_DIRECTORY -v VERBOSE```

For example:

```python scripts/test.py -d data/preprocessed/test -m GIN -lf BCE -md output/train -n 10 -o output/test -v INFO```

#### Predicting SoMs

To predict the Sites of Metabolism of one or multiple molecules with an ensemble classifier consisting of the *n* best trained models run:

```python scripts/predict.py -d DATA_DIRECTORY -m MODEL -lf LOSS_FUNCTION -md MODELS_DIRECTORY -n NUMBER_MODELS -o OUTPUT_DIRECTORY -v VERBOSE```

For example:

```python scripts/predict.py -d data/preprocessed/test -m GIN -lf BCE -md output/train -n 10 -o output/predict -v INFO```

### License

This project is licensed under the MIT license.

### Copyright

Copyright (c) 2023, Roxane Jacob
