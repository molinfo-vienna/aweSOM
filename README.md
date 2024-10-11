aweSOM
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/som_gnn/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/som_gnn/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/SOM_GNN/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/SOM_GNN/branch/main)


A GNN model for the prediction of sites of metabolism (SOMs) in xenobiotics.

### Installation

1. Clone the repository and cd into the repository root:

```git clone https://github.com/molinfo-vienna/aweSOM.git```

```cd aweSOM```

2. Create a conda environment with the required dependencies:

```conda env create --name awesom-env python=3.10```

3. Activate the environment:

```conda activate awesom-env```

4. Install awesom package with ```pip install -e .```


### Usage

#### Determine optimal architecture and hyperparameters via k-fold cross validation:

```python scripts/cv_hp_search.py -i INPUT_PATH -o OUTPUT_PATH -m MODEL -e EPOCHS -n NUM_FOLDS -t NUM_TRIALS```

```INPUT_PATH```: The path to the input data. For model training, only ```.sdf``` input files are currently supported. Please place your file into a subfolder named ```raw/```. Example: the path to the input data is ```data/train/raw/xxx.sdf```, so ```INPUT_PATH``` should be ```data/train```. Running any script (cv_hp_search.py, train.py, test.py) for the first time will create a processed version of the data and place into ```INPUT_PATH/processed``` directory. If such directory already exists, then the already processed data is used. Note that this processed data is not updated with every run. If you wish to modify the input data for which processed data already exists, delete the processed folder prior to reruning your experiments!

```OUTPUT_PATH```: The desired output's location. The best hyperparameters will be stored in a YAML file. The individual validation metrics of each fold will be stored in a CSV file. The best model's checkpoints will be stored in a directory. The averaged predictions made with the best hyp.erparameters will be stored in a text file

```MODEL```: The desired model architecture. Choose between ```M1```, ```M2```, ```M3```, ```M4```, ```M7```, ```M9```, ```M11```, ```M12```.  Default is ```M7```.

* ```M1```: GINConv (https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.conv.GINConv.html)
* ```M2```: GINEConv (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html)
* ```M3```: GATv2Conv (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv)
* ```M4```: MFConv (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv)
* ```M7```: GINEConv + context pooling. This architecture led to the best results on the validation set and was used in the final model.
* ```M9```: GINEConv + molecular features
* ```M11```: GINEConv + DenseNet-inspired skip connections
* ```M12```: GINEConv + ResNet inspired skip connections

```EPOCHS```: The maximum number of training epochs. Default is ```1000```.

```NUM_FOLDS```: The number of cross-validation folds. Default is ```10```.

```NUM_TRIALS```: The number of Optuna trials. Default is ```20```.

Example:

```python scripts/cv_hp_search.py -i /data/train -o output/M7 -m M7 -e 1000 -n 10 -t 20```

#### Model Training

```python scripts/train.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH -m MODEL -e EPOCHS```

```INPUT_PATH```: The path to the input data. For model training, only ```.sdf``` input files are currently supported. Please place your file into a subfolder named ```raw/```. Example: the path to the input data is ```data/train/raw/xxx.sdf```, so ```INPUT_PATH``` should be ```data/train```. Running any script (cv_hp_search.py, train.py, infer.py) for the first time will create a processed version of the data and place into ```INPUT_PATH/processed``` directory. If such directory already exists, then the already processed data is used. Note that this processed data is not updated with every run. If you wish to modify the input data for which processed data already exists, delete the processed folder prior to reruning your experiments!

```CHECKPOINTS_PATH```: The path to the yaml file containing the hyperparameters that were previously determined by running the cv_hp_search.py script on the training data.

```OUTPUT_PATH```: The desired output's location. The best hyperparameters will be stored in a YAML file. The individual validation metrics of each fold will be stored in a CSV file. The best model's checkpoints will be stored in a directory. The averaged predictions made with the best hyp.erparameters will be stored in a text file

```MODEL```: The desired model architecture. Choose between ```M1```, ```M2```, ```M3```, ```M4```, ```M7```, ```M9```, ```M11```, ```M12```.   Default is ```M7```.

```EPOCHS```: The maximum number of training epochs.  Default is ```1000```.

Example:

```python scripts/train.py -i data/train -c output/M7 -o output/M7/ensemble -m M7 -e 1000```

#### Model testing (predicting SoMs for labeled data)

To predict the SoMs of one or multiple *labeled* molecules and output the predictions and the performance metrics run:

```python scripts/test.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH -m test```

```INPUT_PATH```: The path to the input data. For model testing, only .sdf files are currently supported. Please place your file into a subfolder named ```raw/```. Example: the path to the input data is ```data/test/raw/xxx.sdf```, so ```INPUT_PATH``` should be ```data/test```. Running any script (cv_hp_search.py, train.py, infer.py) for the first time will create a processed version of the data and place into ```INPUT_PATH/processed``` directory. If such directory already exists, then the already processed data is used. Note that this processed data is not updated with every run. If you wish to modify the input data for which processed data already exists, delete the processed folder prior to reruning your experiments!

```CHECKPOINTS_PATH```: The path to the trained ensemble model's checkpoints.

```OUTPUT_PATH```: The desired output's location. The performance metrics are written to ```results.txt``` and the individual predictions to ```results.csv```.

Example:

```python scripts/infer.py -i data/test -c output/M7/ensemble -o output/M7/test -m test```

#### Inference (predicting SoMs for unlabeled data)

To predict the SoMs of one or multiple *unlabeled* molecules and output the predictions run:

```python scripts/test.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH -m infer```

```INPUT_PATH```: The path to the input data. For inference, both .sdf and .smi files and are currently supported. Please and place your data into a subfolder named ```raw/```. Example: the data input path is ```data/fipronil/raw/data.smi```, so ```INPUT_PATH``` should be ```data/fipronil```. Running any script (cv_hp_search.py, train.py, infer.py) for the first time will create a processed version of the data and place into ```INPUT_PATH/processed``` directory. If such directory already exists, then the already processed data is used. Note that this processed data is not updated with every run. If you wish to modify the input data for which processed data already exists, delete the processed folder prior to reruning your experiments!

```CHECKPOINTS_PATH```: The path to the trained ensemble model's checkpoints.

```OUTPUT_PATH```: The desired output's location. The individual predictions to ```results.csv```.

Example:

```python scripts/test.py -i data/abemaciclib -c output/M7/ensemble -o output/M7/abemaciclib -m infer```

### License

This project is licensed under the MIT license.
