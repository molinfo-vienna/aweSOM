
import argparse
import os
import ast
import logging
import numpy as np
from rdkit.Chem import PandasTools
import networkx as nx
from torch_geometric import seed_everything

from som_gnn.process_input_data import compute_node_features_matrix, mol_to_nx
from som_gnn.utils import seed_everything



def run(file, split):
    """Computes and saves the necessary data (graph, features, labels, graph_ids)
    to create a PyTorch Geometric custom dataset from an SDF file containing molecules.

    Args:
        dir (string): the directory where the input data is stored
        file (string): the name of the input data file (must be .sdf)
    """
    # Import data from sdf file
    df = PandasTools.LoadSDF(os.path.join(file), removeHs=True)
    df["soms"] = df["soms"].map(ast.literal_eval)

    # Generate networkx graphs from mols and save them in a json file
    df["G"] = df.apply(lambda x: mol_to_nx(x.mol_id, x.ROMol, x.soms), axis=1)
    G = nx.disjoint_union_all(df["G"].to_list())
    # with open(os.path.join(dir, "graph.json"), "w") as f:
    #     f.write(json.dumps(json_graph.node_link_data(G)))

    # Generate and save list of mol ids
    mol_ids = []
    for i in range(len(G.nodes)):
        mol_ids.append(G.nodes[i]["mol_id"])
    mol_ids = np.array(mol_ids)
    # np.save(os.path.join(dir, "mol_ids.npy"), mol_ids)

    # Generate and save list of atom ids
    atom_ids = []
    for i in range(len(G.nodes)):
        atom_ids.append(G.nodes[i]["atom_id"])
    atom_ids = np.array(atom_ids)
    # np.save(os.path.join(dir, "atom_ids.npy"), atom_ids)

    # Generate and save list of labels
    labels = []
    for i in range(len(G.nodes)):
        labels.append(int(G.nodes[i]["is_som"]))
    labels = np.array(labels)
    # np.save(os.path.join(dir, "labels.npy"), labels)

    # Compute node features matrix and save it to node_features.npy
    node_features = compute_node_features_matrix(G)
    # np.save(os.path.join(dir, "node_features.npy"), node_features)

    # df = pd.DataFrame(node_features)
    # corr_matrix = df.corr()
    # plt.imshow(corr_matrix, cmap="binary")
    # plt.savefig(os.path.join(dir, "correlation_matrix.png"))


if __name__ == "__main__":
    
    seed_everything(42)
    
    parser = argparse.ArgumentParser("Preprocess the data.")

    parser.add_argument("-f",
        "--file",
        type=str,
        required=True,
        help="the name of the input data file (must be .sdf)",    
    )
    parser.add_argument("-s",
        "--split",
        type=int,
        required=True,
        help="how much of the input data should be saved as test data."+
        " Maximum value is 100, which means that 100\% of the data will be stored in the test folder under the"+
        " directory in '--dir'.",    
    )
    parser.add_argument("-v",
        "--verbose",
        dest="verbosityLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help="Set the verbosity level of the logger - default is on WARNING."
        )
        

    args = parser.parse_args()

    logging.basicConfig(filename='./output/logfile_preprocess.log', 
                    level=getattr(logging, args.verbosityLevel), 
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    logging.info("Start preprocessing...")

    try:
        run(args.file, args.split)
    except Exception as e:
        logging.error("The preprocess was terminated:",e)