import argparse
import ast
import distutils
import logging
import os
import pandas as pd
import shutil

import CDPL.Chem as Chem

from rdkit.Chem import PandasTools
from torch_geometric import seed_everything

from awesom.process_input_data import generate_preprocessed_data, save_preprocessed_data, load_mol_input
from awesom.utils import seed_everything

def run(dir, file, numberWorkers, predict, RD=False):
    """Computes and saves the necessary data (graph, features, labels, graph_ids)
    to create a PyTorch Geometric custom dataset from an SDF file containing molecules.

    Args:
        dir (string):       the directory where the input data is stored
        file (string):      the name of the data file (with file name extension)
        datType (str):      the encoding of the input data (sdf, smiles or inchi)
        predict (bool):     is the data for prediction purposes?
        numberWorker (int): the number of worker for parallelization
        RD (bool):          If the calculations should be done via RDKit or CDPKit (default)
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

    _, file_extension = os.path.splitext(file)

    ##### generate df

    if RD:
        if file_extension == ".sdf":
            df = PandasTools.LoadSDF(os.path.join(dir, file), removeHs=True)
        elif file_extension == ".smiles":
            df = pd.read_csv(os.path.join(dir, file), names=["smiles"]) 
            PandasTools.AddMoleculeColumnToFrame(df, "smiles")
        else: raise NotImplementedError(f"Invalid file extension: {file_extension}")
    else:
        df = pd.DataFrame()
        mol_list, smiles_list = load_mol_input(str(os.path.join(dir, file)),True)
        df['mol'] = mol_list
        df['smiles'] = smiles_list
        
    if predict:
        df["soms"] = "[0]"

    df["soms"] = df["soms"].map(ast.literal_eval)
    df["ID"] = df.index
    
    print("Preprocessing... This can take a few minutes.")
    logging.info("START preprocessing")

    G, mol_ids, atom_ids, labels, node_features = generate_preprocessed_data(df, numberWorkers, RD)
    logging.info("Saving preprocessed test set...")
    save_preprocessed_data(G, mol_ids, atom_ids, labels, node_features, os.path.join(dir, "preprocessed"))

    print("Preprocessing sucessful!")
    logging.info("END preprocessing")
    
if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Preprocess the data.")

    parser.add_argument("-d",
        "--dir",
        type=str,
        required=True,
        help="The directory where the input data is stored.",    
    )
    parser.add_argument("-f",
        "--file",
        type=str,
        required=True,
        help="The name of the input data file (with file name extension).\
              The file can be either sdf or smiles. E.g., \"data.sdf\".",    
    )
    parser.add_argument("-w",
        "--numberWorkers",
        type=int,
        required=True,
        help="The number of parallel workers. Please note that -w should not \
            be set to a number greater than the number of molecules in the data.",    
    )
    parser.add_argument("-p",
        "--predict",
        type=lambda x:bool(distutils.util.strtobool(x)),
        required=True,
        help="Set to False if the data serves as training data (known SoMs).\
              Set to True if the purpose is to predict the SoMs.",    
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
        run(args.dir,
            args.file, 
            args.numberWorkers,
            args.predict,
            )
    except Exception as e:
        logging.error("The preprocess was terminated:", e)