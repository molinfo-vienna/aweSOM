aweSOM
==============================

A model predicting sites-of-metabolism (SOMs) in xenobiotics with aleatoric and epistemic uncertainty estimation.

Preprint: [https://chemrxiv.org/engage/chemrxiv/article-details/6703a2a851558a15ef56fbea](https://chemrxiv.org/engage/chemrxiv/article-details/67ee93de6dde43c908425567)

⚠️ **Important Notice (June 10, 2025)**

The repository history was rewritten to remove large, outdated folders.  
If you cloned this repository **before June 10, 2025**, please:

1. Delete your local copy

2. Re-clone the repository.

# Installation

1. Clone the repository and cd into the repository root:

```git clone https://github.com/molinfo-vienna/aweSOM.git```

```cd aweSOM```

2. Create a conda environment with the required python version:

```conda create --name awesom-env python=3.10```

3. Activate the environment:

```conda activate awesom-env```

4. Install awesom package with ```pip install -e .```


# Data
The use of the trained aweSOM model requires a license for the MetaQSAR database (10.1021/acs.jmedchem.7b01473). To demonstrate the validity of the software presented in this repository, we trained and tested an example version of aweSOM using public metabolism data, commonly referred to as the Zaretzki data set. The original data can be found at 10.1021/ci300009z. The curated data used for demonstrating how to train and test aweSOM, as well as the trained example models and outputs can be found at https://figshare.com/s/9fb1b972d390d8f0e16a.


# Usage

## Inference (predicting SOMs for unlabeled data)

To predict the SOMs of one or multiple *unlabeled* molecules and output the predictions run:

```python scripts/test.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH -m infer```

```INPUT_PATH```: The path to the input data. For inference, both SD-files (.sdf) and smiles files (.smi, .smiles) and are currently supported. Running test.py will create processed data files and place them into ```INPUT_PATH/processed/```.

```CHECKPOINTS_PATH```: The path to the trained ensemble model's checkpoints.

```OUTPUT_PATH```: The desired output's location. The individual predictions to ```results.csv```.

Example:

```python scripts/test.py -i data/case_studies -c output/model -o output/case_studies_by_example_model -m infer```

## Model re-training and testing

### 1. Determine optimal architecture and hyperparameters via k-fold cross validation:

```python scripts/cv_hp_search.py -i INPUT_PATH -o OUTPUT_PATH -m MODEL -e EPOCHS -n NUM_CV_FOLDS -t NUM_TRIALS```

```INPUT_PATH```: The folder holding the input data file. Only ```.sdf``` input files are currently supported. Running cv_hp_search.py will create processed data files and place them into ```INPUT_PATH/processed/```.

```OUTPUT_PATH```: The desired output's location. The best hyperparameters will be stored in a YAML file (best_hparams.yaml). The individual validation metrics of each fold will be stored in CSV files (validation_foldX.csv). The best model's checkpoints will be stored in a directory (logs). The averaged predictions made with the best hyperparameters will be stored in a text file (validation.txt).

```MODEL```: The desired model architecture. Choose between ```M1```, ```M2```, ```M3```, ```M4```, ```M7```, ```M9```, ```M11```, ```M12```.  Default is ```M7```.

* ```M1```: GINConv (https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.conv.GINConv.html)
* ```M2```: GINEConv (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html)
* ```M3```: GATv2Conv (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv)
* ```M4```: MFConv (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MFConv.html#torch_geometric.nn.conv.MFConv)
* ```M7```: GINEConv + context pooling. This architecture led to the best results on the validation set and was used in the final model.
* ```M9```: GINEConv + molecular features
* ```M11```: GINEConv + DenseNet-inspired skip connections
* ```M12```: GINEConv + ResNet inspired skip connections

```EPOCHS```: The maximum number of training epochs. Default is ```500```.

```NUM_CV_FOLDS```: The number of cross-validation folds. Default is ```10```.

```NUM_TRIALS```: The number of Optuna trials. Default is ```20```.

Example:

```python scripts/cv_hp_search.py -i /data/train -o output/cv_hp_search```

### 2. Model Training

```python scripts/train.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH```

```INPUT_PATH```: The path to the input data. Only ```.sdf``` input files are currently supported. Running train.py will create processed data files and place them into ```INPUT_PATH/processed/```.

```CHECKPOINTS_PATH```: The path to the yaml file containing the hyperparameters that were previously determined by running the cv_hp_search.py script on the training data.

```OUTPUT_PATH```: The desired output's location. Per default, training generates an ensemble of 10 models.

Example:

```python scripts/train.py -i data/train -c output/cv_hp_search -o output/model```

### 3. Model testing (predicting SOMs for labeled data)

To predict the SOMs of one or multiple *labeled* molecules and output the predictions and the performance metrics run:

```python scripts/test.py -i INPUT_PATH -c CHECKPOINTS_PATH -o OUTPUT_PATH -m test```

```INPUT_PATH```: The path to the input data. For model testing, only .sdf files are currently supported.  Running test.py will create processed data files and place them into ```INPUT_PATH/processed/```.

```CHECKPOINTS_PATH```: The path to the trained ensemble model's checkpoints.

```OUTPUT_PATH```: The desired output's location. The performance metrics are written to ```results.txt``` and the individual predictions to ```results.csv```.

Example:

```python scripts/test.py -i data/test -c output/model -o output/test -m test```

# License

This project is licensed under the MIT license.
