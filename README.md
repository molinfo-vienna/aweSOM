SOM-GNN
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/som_gnn/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/som_gnn/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/SOM_GNN/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/SOM_GNN/branch/main)


A Graph Neural Network (GNN) for the prediction of Sites of Metabolism (SoMs).

### Installation

1. Clone the repository and cd into the repository root:

```git clone https://github.com/molinfo-vienna/som_gnn.git```

```cd som_gnn```

2. Create a conda environment with the required dependencies:

```conda env create -f som-gnn-env.yml```

3. Activate the environment:

```conda activate som-gnn-env```

### Usage

#### Dataset Generation

To load new data for model training/testing and/or prediction purposes run the following:

```python scripts/preprocess.py -f FILE.sdf -d DIRECTORY -s SPLIT -v VERBOSE```

```FILE``` is the name of the data file. Only ```.sdf``` files are currently supported.
```DIRECTORY``` is the directory under which the data file is stored.
```SPLIT``` indicates the split ratio of training and test data. E.g. 20 means that 20% of the data will be stored as test data (recommended for training/testing purposes). We recommend setting ```SPLIT``` to 100 for predicting purposes, i.e. if you simply wish to apply the already trained model to your data.
```VERBOSE``` should be set to the desired verbosity level, e.g. ```INFO```.

The output of the preprocessing steps will be written in the ```DIRECTORY/preprocessed/train/``` and ```DIRECTORY/preprocessed/test/``` folders. These folders will be automatically created when executing ```preprocess.py```.

#### Model Training

To train a model with a specific set of hyperparameters run:

```python scripts/train -d DATA_DIRECTORY -hd DIMENSION_HIDDEN_LAYERS -do DROPOUT -e EPOCHS -lr LEARNING_RATE -wd WEIGHT_DECAY -bs BATCH_SIZE -p PATIENCE -dt DELTA -o OUTPUT_DIRECTORY -v VERBOSE```

For example:

```python scripts/train.py -d data/preprocessed/train -hd 32 -do 0.2 -e 1000 -lr 0.001 -wd 0.001 -bs 16 -p 20 -dt 0 -o output/train -v INFO```

#### Model Testing

To test the performance of an ensemble classifier consisting of the *n* best trained models run:

```python scripts/test -d DATA_DIRECTORY -m MODELS_DIRECTORY -n NUMBER_MODELS -o OUTPUT_DIRECTORY -v VERBOSE```

For example:

```python scripts/test.py -d data/preprocessed/test -m output/train -n 10 -o output/test -v INFO```

#### Predicting SoMs

To predict the Sites of Metabolism of one or multiple molecules with an ensemble classifier consisting of the *n* best trained models run:

```python scripts/predict -d DATA_DIRECTORY -m MODELS_DIRECTORY -n NUMBER_MODELS -o OUTPUT_DIRECTORY -v VERBOSE```

For example:

```python scripts/predict.py -d data/preprocessed/test -m output/train -n 10 -o output/predict -v INFO```

### License

This project is licensed under the MIT license.

### Copyright

Copyright (c) 2023, Roxane Jacob
