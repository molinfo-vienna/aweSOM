import ast
import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm

def mol_to_nx(mol_id, mol, soms):
    """Takes as input an RDKit mol object and return its corresponding NetworkX Graph.

    Args:
        mol (RDKit mol object)

    Returns:
        G (NetworkX.Graph)
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node( atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    degree=atom.GetDegree(),
                    formal_charge=atom.GetFormalCharge(),
                    hybridization=atom.GetHybridization(),
                    num_hs=atom.GetTotalNumHs(),
                    is_in_ring=atom.IsInRing(),
                    is_aromatic=atom.GetIsAromatic(),
                    vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()),
                    covalent_radius = Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()),
                    mol_id=int(mol_id),
                    is_som=(atom.GetIdx() in soms))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx())
    return G


def compute_features_tensor(G):

    # get features dimension:
    num_nodes = len(G.nodes)
    num_features=len(G.nodes[0])-2

    # construct features matrix of shape (number of atoms, number of features)
    features = np.zeros((num_nodes, num_features))

    # write features to features matrix
    for i in tqdm(range(num_nodes)):
        current_node = G.nodes[i]

        atomic_num = [current_node['atomic_num']]
        degree = [current_node['degree']]
        formal_charge = [current_node['formal_charge']]
        hybridization = [current_node['hybridization']]
        num_hs = [current_node['num_hs']]
        is_in_ring = [int(current_node['is_in_ring'])]
        is_aromatic = [int(current_node['is_aromatic'])]
        vdw_radius = [current_node['vdw_radius']]
        covalent_radius = [current_node['covalent_radius']]

        node_features_vector = atomic_num + degree + formal_charge + hybridization + num_hs + is_in_ring + is_aromatic + vdw_radius + covalent_radius

        features[i,:] = np.array(node_features_vector)

    return features


def process_data(path):
    """Computes and saves the necessary data (graph, features, labels, graph_ids)
    to create a PyTorch Geometric custom dataset from an SDF file containing molecules.

    Args:
        path (string): the path where the SDF data is stored.
    """
    # Import data from sdf file
    df = PandasTools.LoadSDF(path, removeHs=True)
    df['soms_new'] = df['soms_new'].map(ast.literal_eval)

    # Generate networkx graphs from mols and save them in a json file
    df["G"] = df.apply(lambda x: mol_to_nx(x.mol_id, x.ROMol, x.soms_new), axis=1)
    G = nx.disjoint_union_all(df["G"].to_list())
    with open('data/graph.json', 'w') as f:
            f.write(json.dumps(json_graph.node_link_data(G)))

    # Generate and save list of labels
    labels = []
    for i in range(len(G.nodes)):
        labels.append(int(G.nodes[i]['is_som']))
    labels = np.array(labels)
    np.save('data/labels.npy', labels)

    # Generate and save list of mol ids
    mol_ids = []
    for i in range(len(G.nodes)):
        mol_ids.append(G.nodes[i]['mol_id'])
    mol_ids = np.array(mol_ids)
    np.save('data/mol_ids.npy', mol_ids)

    # Compute features matrix and save it to features.npy
    features = compute_features_tensor(G)
    np.save('data/features.npy', features)
