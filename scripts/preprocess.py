import argparse
import ast
import distutils.util
import logging
import os
import pandas as pd
import shutil

from rdkit.Chem import PandasTools

from awesom.process_input_data import generate_preprocessed_data, save_preprocessed_data
from awesom.utils import seed_everything


if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser("Preprocess the data.")

    parser.add_argument(
        "-i",
        "--inputPath",
        type=str,
        required=True,
        help='The absolute path to the input data file (with file name extension).\
              The file can be either sdf or smiles. E.g., "data.sdf".',
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        type=str,
        required=True,
        help="The directory of the preprocessed data.",
    )
    parser.add_argument(
        "-w",
        "--numberWorkers",
        type=int,
        required=True,
        help="The number of parallel workers. Please note that -w should not \
            be set to a number greater than the number of molecules in the data.",
    )
    parser.add_argument(
        "-p",
        "--predict",
        type=lambda x: bool(distutils.util.strtobool(x)),
        required=True,
        help="Set to False if the data serves as training data (known SoMs).\
              Set to True if the purpose is to predict the SoMs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosityLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the verbosity level of the logger - default is on INFO.",
    )

    args = parser.parse_args()

    if os.path.exists(args.outputDir):
        overwrite = input("Folder already exists. Overwrite? [y/n] \n")
        if overwrite == "y":
            shutil.rmtree(args.outputDir)
            os.makedirs(args.outputDir)
        if overwrite == "n":
            raise FileExistsError("Folder already exists.")
    else:
        os.makedirs(args.outputDir)

    logging.basicConfig(
        filename=os.path.join(args.outputDir, "logfile_preprocess.log"),
        level=getattr(logging, args.verbosityLevel),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    _, file_extension = os.path.splitext(args.inputPath)

    if file_extension == ".sdf":
        df = PandasTools.LoadSDF(args.inputPath, removeHs=True)
    elif file_extension == ".smiles":
        df = pd.read_csv(args.inputPath, names=["smiles"])
        PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    else:
        raise NotImplementedError(f"Invalid file extension: {file_extension}")

    if args.predict:
        df["soms"] = "[]"

    df["soms"] = df["soms"].map(ast.literal_eval)
    df["ID"] = df.index

    print("Preprocessing... This can take a few minutes.")
    logging.info("START preprocessing")

    G, mol_ids, atom_ids, labels, node_features = generate_preprocessed_data(
        df, args.numberWorkers
    )
    logging.info("Saving preprocessed test set...")
    save_preprocessed_data(G, mol_ids, atom_ids, labels, node_features, args.outputDir)

    print("Preprocessing sucessful!")
    logging.info("END preprocessing")
