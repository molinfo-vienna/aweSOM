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
        mol (RDKit Mol object)

    Returns:
        G (NetworkX Graph object)
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node( atom.GetIdx(),
                    atomic_num = atom.GetAtomicNum(),
                    mass = atom.GetMass(),
                    degree = atom.GetTotalDegree(),
                    valence = atom.GetTotalValence(),
                    formal_charge = atom.GetFormalCharge(),
                    hybridization = atom.GetHybridization(),
                    num_hs = atom.GetTotalNumHs(),
                    is_in_ring_3 = atom.IsInRingSize(3),
                    is_in_ring_4 = atom.IsInRingSize(4),
                    is_in_ring_5 = atom.IsInRingSize(5),
                    is_in_ring_6 = atom.IsInRingSize(6),
                    is_in_ring_7 = atom.IsInRingSize(7),
                    is_in_ring_8 = atom.IsInRingSize(8),
                    is_aromatic = atom.GetIsAromatic(),
                    vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()),
                    covalent_radius = Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()),
                    # the next two elements are later used to compute the labels but will of course
                    # not be used as features!
                    mol_id = int(mol_id),
                    is_som = (atom.GetIdx() in soms))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx())
    return G


def compute_features_matrix(G):
    """Takes as input a NetworkX Graph object (which already contains the 
    features for each individual nodes) and extracts/returns its corresponding feature matrix.

    Args:
        G (NetworkX Graph object)

    Returns:
        _features (numpy array): a numpy array of dimension (number_of_nodes, number_of_features)
    """

    # get features dimension:
    num_nodes = len(G.nodes)
    num_features=len(G.nodes[0])-2

    # construct features matrix of shape (number of atoms, number of features)
    features = np.zeros((num_nodes, num_features))

    # write features to features matrix
    for i in tqdm(range(num_nodes)):
        current_node = G.nodes[i]

        atomic_num = [current_node['atomic_num']]
        mass = [current_node['mass']]
        degree = [current_node['degree']]
        valence = [current_node['valence']]
        formal_charge = [current_node['formal_charge']]
        hybridization = [current_node['hybridization']]
        num_hs = [current_node['num_hs']]
        is_in_ring_3 = [int(current_node['is_in_ring_3'])]
        is_in_ring_4 = [int(current_node['is_in_ring_4'])]
        is_in_ring_5 = [int(current_node['is_in_ring_5'])]
        is_in_ring_6 = [int(current_node['is_in_ring_6'])]
        is_in_ring_7 = [int(current_node['is_in_ring_7'])]
        is_in_ring_8 = [int(current_node['is_in_ring_8'])]
        is_aromatic = [int(current_node['is_aromatic'])]
        vdw_radius = [current_node['vdw_radius']]
        covalent_radius = [current_node['covalent_radius']]

        node_features_vector = atomic_num + mass + degree + valence + formal_charge + \
            hybridization + num_hs + is_in_ring_3 + is_in_ring_4 + is_in_ring_5 + \
                is_in_ring_6 + is_in_ring_7 + is_in_ring_8 +is_aromatic + vdw_radius + \
                    covalent_radius

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
    features = compute_features_matrix(G)
    np.save('data/features.npy', features)
