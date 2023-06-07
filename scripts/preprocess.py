import argparse
import ast
import logging
import os
import shutil

from rdkit.Chem import PandasTools
from torch_geometric import seed_everything

from awesom.process_input_data import generate_preprocessed_data, save_preprocessed_data
from awesom.utils import seed_everything, make_dir

def run(file, dir, split, featuresCombination):
    """Computes and saves the necessary data (graph, features, labels, graph_ids)
    to create a PyTorch Geometric custom dataset from an SDF file containing molecules.

    Args:
        file (string):  the name of the (.sdf) input data file
        dir (string):   the directory where the input data is stored
        split (int):    the split ratio of train/test set (e.g 20 means 
                        that 20% of the data is in the test set)
        featuresCombination (string): the desired featurization scheme
    """

    # Import data
    df = PandasTools.LoadSDF(os.path.join(dir, file), removeHs=True)
    df["soms"] = df["soms"].map(ast.literal_eval)

    # Create *dir*/preprocessed/train/ and dir/preprocessed/test/ directories
    if os.path.exists(os.path.join(dir, "preprocessed")):
        overwrite = input("Folder already exists. Overwrite? [y/n] \n")
        if overwrite == "y":
            shutil.rmtree(os.path.join(dir, "preprocessed"))
            make_dir(dir)
        if overwrite == "n":
            return None
    else:
        make_dir(dir)

    print("Preprocessing... This can take a few minutes.")

    # Split the data into train/test set according to the split ratio
    # Note: df.sample shuffles dataframe randomly before sampling
    if split > 0:
        df_test = df.sample(frac = split/100)
        logging.info("Start preprocessing test set...")
        # Generate and save preprocessed data
        # under *dir*/processed/train or test folder
        G_test, mol_ids_test, atom_ids_test, labels_test, node_features_test = generate_preprocessed_data(df_test, featuresCombination)
        logging.info("Saving preprocessed test set...")
        save_preprocessed_data(G_test, mol_ids_test, atom_ids_test, labels_test, node_features_test, os.path.join(dir, "preprocessed/test"))
        logging.info("Preprocessing test set sucessful!")
    if split != 100:
        logging.info("Start preprocessing training set...")
        if split == 0:
            df_train = df
        else:
            df_train = df.drop(df_test.index)
        G_train, mol_ids_train, atom_ids_train, labels_train, node_features_train = generate_preprocessed_data(df_train, featuresCombination)
        logging.info("Saving preprocessed train set...")
        save_preprocessed_data(G_train, mol_ids_train, atom_ids_train, labels_train, node_features_train , os.path.join(dir, "preprocessed/train"))
        logging.info("Preprocessing training set sucessful!")

    print("Preprocessing sucessful!")

if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Preprocess the data.")

    parser.add_argument("-f",
        "--file",
        type=str,
        required=True,
        help="The name of the input data file (must be .sdf).",    
    )
    parser.add_argument("-d",
        "--dir",
        type=str,
        required=True,
        help="The directory where the input data is stored.",    
    )
    parser.add_argument("-s",
        "--split",
        type=int,
        required=True,
        help="The split ratio of train/test set (e.g 20 means that 20% of the data is in the test set)",  
    )
    parser.add_argument("-fc",
        "--featuresCombination",
        type=str,
        required=True,
        help="The desired featurization scheme. Choose between \'FC1\', \'FC2\', \'FC3\' and \'FC4\'.",    
    )
    parser.add_argument("-v",
        "--verbose",
        dest="verbosityLevel", 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the verbosity level of the logger - default is on INFO."
        )

    args = parser.parse_args()

    logging.basicConfig(filename= os.path.join(args.dir, 'logfile_preprocess.log'), 
                    level=getattr(logging, args.verbosityLevel), 
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    try:
        run(args.file, 
            args.dir, 
            args.split, 
            args.featuresCombination, 
            )
    except Exception as e:
        logging.error("The preprocess was terminated:", e)