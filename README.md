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

OR install requirements manually via requirements.txt

### Usage

#### Determine optimal architecture and hyperparameters via k-fold cross validation:

```python scripts/cv_hp_search.py -i INPUT_PATH -o OUTPUT_PATH -m MODEL -e EPOCHS -n NUM_FOLDS -t NUM_TRIALS```

```INPUT_PATH```: The path to the input data. For model training, only ```.sdf``` input files are currently supported. Please place your file into a subfolder named ```raw/```. Example: the path to the input data is ```data/train/raw/xxx.sdf```, so ```INPUT_PATH``` should be ```data/train```. Running any script (cv_hp_search.py, train.py, infer.py) for the first time will create a processed version of the data and place into ```INPUT_PATH/processed``` directory. If such directory already exists, then the already processed data is used. Note that this processed data is not updated with every run. If you wish to modify the input data for which processed data already exists, delete the processed folder prior to reruning your experiments!

```OUTPUT_PATH```: The desired output's location. The best hyperparameters will be stored in a YAML file. The individual validation metrics of each fold will be stored in a CSV file. The best model's checkpoints will be stored in a directory. The averaged predictions made with the best hyp.erparameters will be stored in a text file

```MODEL```: The desired model architecture. Choose between ```M1```, ```M2```, ```M4```, ```M6```, ```M7```, ```M8```, ```M11```, ```M12```, ```M13```, ```M14```.

```EPOCHS```: The maximum number of training epochs.

```NUM_FOLDS```: The number of cross-validation folds.

```NUM_TRIALS```: The number of Optuna trials.

There is the possibility to choose between ```RDKit``` and ```CDPKit``` for preprocessing the input data. ```RDKit``` is set as default, but if you prefer to use ```CDPKit``` then change the ```KIT``` variable in ```dataset.py``` from ```RDKIT``` to ```CDPKIT```.

Example:

```python scripts/cv_hp_search.py -i /data/train -o output/M4 -m M4 -e 1000 -n 10 -t 50```

#### Model Training

```python scripts/train.py -i INPUT_PATH -y HYPERPARAMETERS_YAML_PATH -o OUTPUT_PATH -m MODEL -e EPOCHS -s ENSEMBLE_SIZE```

```INPUT_PATH```: The path to the input data. For model training, only ```.sdf``` input files are currently supported. Please place your file into a subfolder named ```raw/```. Example: the path to the input data is ```data/train/raw/xxx.sdf```, so ```INPUT_PATH``` should be ```data/train```. Running any script (cv_hp_search.py, train.py, infer.py) for the first time will create a processed version of the data and place into ```INPUT_PATH/processed``` directory. If such directory already exists, then the already processed data is used. Note that this processed data is not updated with every run. If you wish to modify the input data for which processed data already exists, delete the processed folder prior to reruning your experiments!

```HYPERPARAMETERS_YAML_PATH```: The path to the yaml file containing the hyperparameters that were previously determined by running the cv_hp_search.py script on the training data.

```OUTPUT_PATH```: The desired output's location. The best hyperparameters will be stored in a YAML file. The individual validation metrics of each fold will be stored in a CSV file. The best model's checkpoints will be stored in a directory. The averaged predictions made with the best hyp.erparameters will be stored in a text file

```MODEL```: The desired model architecture. Choose between ```M1```, ```M2```, ```M4```, ```M6```, ```M7```, ```M8```, ```M11```, ```M12```, ```M13```, ```M14```.

```EPOCHS```: The maximum number of training epochs.

```ENSEMBLE_SIZE```: The desired number of models in the final deep ensemble model.

Example:

```python scripts/train.py -i data/train -y output/M4 -o output/M4/ensemble -m M4 -e 1000 -s 100```

#### Model testing (predicting SoMs for unseen, labeled data)

To predict the SoMs of one or multiple *labeled* molecules and output the predictions and the performance metrics run:

```python scripts/infer.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH -t TEST```

```INPUT_PATH```: The path to the input data. For model testing, only .sdf files are currently supported. Please place your file into a subfolder named ```raw/```. Example: the path to the input data is ```data/test/raw/xxx.sdf```, so ```INPUT_PATH``` should be ```data/test```. Running any script (cv_hp_search.py, train.py, infer.py) for the first time will create a processed version of the data and place into ```INPUT_PATH/processed``` directory. If such directory already exists, then the already processed data is used. Note that this processed data is not updated with every run. If you wish to modify the input data for which processed data already exists, delete the processed folder prior to reruning your experiments!

```CHECKPOINTS_PATH```: The path to the trained ensemble model's checkpoints.

```OUTPUT_PATH```: The desired output's location. The performance metrics are written to ```results.txt``` and the individual predictions to ```results.csv```.

```TEST```: Whether to perform inference on non-labeled data (default: False) or testing on labeled data (True). If set to true, the script assumes that true labels are provided and computes the classification metrics (MCC, precision, recall, top2 correctness rate, atomic and molecular AUROCs, and atomic and molecular R-precisions), which are then written to ```OUTPUT_PATH/results.txt```.

Example:

```python scripts/infer.py -i data/test -c output/M4/ensemble/ -o output/M4/test -t```

#### Inference (predicting SoMs for unseen, unlabeled data)

To predict the SoMs of one or multiple *unlabeled* molecules and output the predictions run:

```python scripts/infer.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH```

```INPUT_PATH```: The path to the input data. For inference, both .sdf and .smi files and are currently supported. Please and place your data into a subfolder named ```raw/```. Example: the data input path is ```data/fipronil/raw/data.smi```, so ```INPUT_PATH``` should be ```data/fipronil```. Running any script (cv_hp_search.py, train.py, infer.py) for the first time will create a processed version of the data and place into ```INPUT_PATH/processed``` directory. If such directory already exists, then the already processed data is used. Note that this processed data is not updated with every run. If you wish to modify the input data for which processed data already exists, delete the processed folder prior to reruning your experiments!

```CHECKPOINTS_PATH```: The path to the trained ensemble model's checkpoints.

```OUTPUT_PATH```: The desired output's location. The individual predictions to ```results.csv```.

Example:

```python scripts/infer.py -i data/fipronil -c output/M4/ensemble/ -o output/M4/fipronil```

### License

This project is licensed under the MIT license.
