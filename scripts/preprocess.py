import argparse
import ast
import logging
import os
import shutil

from rdkit.Chem import PandasTools
from torch_geometric import seed_everything

from som_gnn.process_input_data import generate_preprocessed_data, save_preprocessed_data
from som_gnn.utils import seed_everything, make_dir

def run(file, dir, split):
    """Computes and saves the necessary data (graph, features, labels, graph_ids)
    to create a PyTorch Geometric custom dataset from an SDF file containing molecules.

    Args:
        dir (string): the directory where the input data is stored
        file (string): the name of the input data file (must be .sdf)
        split (int): the split ratio of train/test set n(e.g 20 means that 
                20% of the data are in the test set.)
    """

    # Import data from sdf file
    df = PandasTools.LoadSDF(os.path.join(dir, file), removeHs=True)
    df["soms"] = df["soms"].map(ast.literal_eval)

    # Check whether the directory data/processed/*input_sdf_file_name*/ already exists,
    # and add a prompt if the folder exists and the user wants to override it.
    # Otherwise just create the train and test folders under that directory.
    if os.path.exists(os.path.join(dir, os.path.splitext(file)[0])):
        overwrite = input("Folder already exists. Overwrite? [y/n] \n")
        if overwrite == "y":
            shutil.rmtree(os.path.join(dir, os.path.splitext(file)[0]))
            make_dir(file, dir)
        if overwrite == "n":
            return None
    else:
        make_dir(file, dir)

    # Split the data into train/test set according to the split ratio
    # Note: df.sample shuffles df randomly before sampling
    # Generate and save preprocessed data
    # under data/processed/*input_sdf_file_name*/train or test folder
    df_test = df.sample(frac = split/100)
    logging.info("Start preprocessing test set...")
    G_test, mol_ids_test, atom_ids_test, labels_test, node_features_test = generate_preprocessed_data(df_test)
    save_preprocessed_data(G_test, mol_ids_test, atom_ids_test, labels_test, node_features_test, os.path.join(dir, os.path.splitext(file)[0], "test"))
    logging.info("Preprocessing test set sucessful!")
    
    if split != 100:
        logging.info("Start preprocessing training set...")
        df_train = df.drop(df_test.index)
        G_train, mol_ids_train, atom_ids_train, labels_train, node_features_train = generate_preprocessed_data(df_train)
        save_preprocessed_data(G_train, mol_ids_train, atom_ids_train, labels_train, node_features_train , os.path.join(dir, os.path.splitext(file)[0], "train"))
        logging.info("Preprocessing training set sucessful!")

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
        help="How much of the input data should be saved as test data.",  
    )
    parser.add_argument("-v",
        "--verbose",
        dest="verbosityLevel", 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help="Set the verbosity level of the logger - default is on WARNING."
        )

    args = parser.parse_args()

    logging.basicConfig(filename= os.path.join(args.dir, 'logfile_preprocess.log'), 
                    level=getattr(logging, args.verbosityLevel), 
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    logging.info("Start preprocessing...")

    try:
        run(args.file, args.dir, args.split)
    except Exception as e:
        logging.error("The preprocess was terminated:", e)