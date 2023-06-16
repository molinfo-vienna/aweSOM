import argparse
import ast
import logging
import os
import shutil

from rdkit.Chem import PandasTools
from torch_geometric import seed_everything

from awesom.process_input_data import generate_preprocessed_data, save_preprocessed_data
from awesom.utils import seed_everything

def run(file, dir, numberWorkers):
    """Computes and saves the necessary data (graph, features, labels, graph_ids)
    to create a PyTorch Geometric custom dataset from an SDF file containing molecules.

    Args:
        file (string):      the name of the (.sdf) input data file
        dir (string):       the directory where the input data is stored
        numberWorker (int): the number of worker for parallelization
    """

    if os.path.exists(os.path.join(dir, "preprocessed")):
        overwrite = input("Folder already exists. Overwrite? [y/n] \n")
        if overwrite == "y":
            shutil.rmtree(os.path.join(dir, "preprocessed"))
            os.mkdir(os.path.join(dir, "preprocessed"))
        if overwrite == "n":
            return None
    else:
        os.mkdir(os.path.join(dir, "preprocessed"))

    # Import data
    df = PandasTools.LoadSDF(os.path.join(dir, file), removeHs=True)
    df["soms"] = df["soms"].map(ast.literal_eval)

    print("Preprocessing... This can take a few minutes.")
    logging.info("START preprocessing")
    # Generate and save preprocessed data under *dir* folder
    G, mol_ids, atom_ids, labels, node_features = generate_preprocessed_data(df, numberWorkers)
    logging.info("Saving preprocessed test set...")
    save_preprocessed_data(G, mol_ids, atom_ids, labels, node_features, os.path.join(dir, "preprocessed"))
    print("Preprocessing sucessful!")
    logging.info("END preprocessing")
    
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
    parser.add_argument("-w",
        "--numberWorkers",
        type=int,
        required=True,
        help="The number of parallel workers. Please note that -w should not \
            be set to a number greater than the number of molecules in the data.",    
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
            args.numberWorkers,
            )
    except Exception as e:
        logging.error("The preprocess was terminated:", e)