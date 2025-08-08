aweSOM
==============================

A model predicting sites-of-metabolism (SOMs) in xenobiotics with aleatoric and epistemic uncertainty estimation.

Paper: https://pubs.acs.org/doi/10.1021/acs.jcim.5c00762

⚠️ **Important Notice (June 10, 2025)**

The repository history was rewritten to remove large, outdated folders. If you cloned this repository **before June 10, 2025**, please:

1. Delete your local copy

2. Re-clone the repository

# Installation

1. Clone the repository and cd into the repository root:

```git clone https://github.com/molinfo-vienna/aweSOM.git```

```cd aweSOM```

2. Create a conda environment with the required python version:

```conda create --name awesom-env python=3.10```

3. Activate the environment:

```conda activate awesom-env```

4. Install awesom package with ```pip install .```


# Data
The use of the trained aweSOM model requires a license for the MetaQSAR database (10.1021/acs.jmedchem.7b01473). To demonstrate the validity of the software presented in this repository, we trained and tested an example version of aweSOM using public metabolism data, commonly referred to as the Zaretzki data set. The original data can be found at 10.1021/ci300009z. The curated data used for demonstrating how to train and test aweSOM, as well as the trained example models and outputs can be found at https://figshare.com/s/9888313140b2e77987b8.


# Usage

## Inference (predicting SOMs for unlabeled data)

To predict the SOMs of one or multiple *unlabeled* molecules and output the predictions run:

```python scripts/test.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH -m infer```

**Arguments:**
- `-i, --input`: The path to the folder containing the input data file. For inference, both SD-files (.sdf) and smiles files (.smi, .smiles) are currently supported. Running test.py will create processed data files and place them into `INPUT_PATH/processed/`.
- `-c, --checkpoints`: The path to the trained ensemble model's checkpoints directory.
- `-o, --output`: The desired output's location. The individual predictions will be saved to `results.csv`.
- `-m, --mode`: Must be set to `infer` for inference mode.

**Example:**
```python scripts/test.py -i data/case_studies -c output/model -o output/case_studies_by_example_model -m infer```

## Model re-training and testing

### 1. Determine optimal architecture and hyperparameters via k-fold cross validation:

```python scripts/cv_hp_search.py -i INPUT_PATH -o OUTPUT_PATH [OPTIONS]```

**Required Arguments:**
- `-i, --input`: The path to the folder containing the input data file. Only `.sdf` input files are currently supported. Running cv_hp_search.py will create processed data files and place them into `INPUT_PATH/processed/`.
- `-o, --output`: The desired output's location. The best hyperparameters will be stored in `best_hparams.yaml`. The individual validation metrics of each fold will be stored in CSV files (`validation_fold_0.csv`, `validation_fold_1.csv`, etc.). The averaged validation metrics with standard deviations will be stored in `validation.txt`.

**Optional Arguments:**
- `--epochs`: The maximum number of training epochs. Default is `500`.
- `--folds`: The number of cross-validation folds. Default is `10`.
- `--trials`: The number of Optuna trials for hyperparameter optimization. Default is `20`.
- `--batch_size`: The batch size for training. Default is `32`.

**Example:**
```python scripts/cv_hp_search.py -i data/train -o output/cv_hp_search --epochs 300 --folds 5 --trials 15```

### 2. Model Training

```python scripts/train.py -i INPUT_PATH -c CONFIG_PATH -o OUTPUT_PATH [OPTIONS]```

**Required Arguments:**
- `-i, --input`: The path to the folder containing the input data file. Only `.sdf` input files are currently supported. Running train.py will create processed data files and place them into `INPUT_PATH/processed/`.
- `-c, --config`: The path to the directory containing the `best_hparams.yaml` file with hyperparameters that were previously determined by running the cv_hp_search.py script.
- `-o, --output`: The desired output's location. Per default, training generates an ensemble of 10 models.

**Optional Arguments:**
- `--batch_size`: The batch size for training. Default is `32`.
- `--ensemble_size`: The number of models in the ensemble. Default is `10`.

**Example:**
```python scripts/train.py -i data/train -c output/cv_hp_search -o output/model --ensemble_size 15```

### 3. Model testing (predicting SOMs for labeled data)

To predict the SOMs of one or multiple *labeled* molecules and output the predictions and the performance metrics run:

```python scripts/test.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH -m test```

**Arguments:**
- `-i, --input`: The path to the folder containing the input data file. For model testing, only `.sdf` files are currently supported. Running test.py will create processed data files and place them into `INPUT_PATH/processed/`.
- `-c, --checkpoints`: The path to the trained ensemble model's checkpoints directory.
- `-o, --output`: The desired output's location. The performance metrics are written to `results.txt` and the individual predictions to `results.csv`.
- `-m, --mode`: Must be set to `test` for testing mode.

**Example:**
```python scripts/test.py -i data/test -c output/model -o output/test -m test```

## Output Files

### Cross-validation (cv_hp_search.py)
- `best_hparams.yaml`: Best hyperparameters found by Optuna
- `validation_fold_X.csv`: Detailed predictions for each fold (X = 0, 1, 2, ...)
- `validation.txt`: Validation metrics with standard deviations across folds
- `study.db`: Optuna study database for resuming optimization

### Training (train.py)
- `model_X/`: Directory for each model in the ensemble (X = 0, 1, 2, ...)
  - `model_X/checkpoints/best_model.ckpt`: Best model checkpoint
  - `model_X/logs/`: TensorBoard logs
- `seeds.txt`: Random seeds used for each model for reproducibility

### Testing/Inference (test.py)
- `results.csv`: Detailed predictions with uncertainties
- `results.txt`: Performance metrics with bootstrap confidence intervals (for test mode)
- `roc.png`: ROC curve plot (for test mode)

# License

This project is licensed under the MIT license.
