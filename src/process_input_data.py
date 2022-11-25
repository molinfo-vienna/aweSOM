import ast
import json
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, PandasTools, rdMolDescriptors
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def mol_to_nx(mol_id, mol, soms):
    """Takes as input an RDKit mol object and return its corresponding NetworkX Graph.

    Args:
        mol (RDKit Mol object)

    Returns:
        G (NetworkX Graph object)
    """
    G = nx.Graph()

    # Assign each atom its molecular and atomic features and make it a node of G
    # is_som = [(atom_idx in soms) for atom_idx in range(mol.GetNumAtoms())]
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        G.add_node( atom_idx, # node identifier
                    # atom features
                    atomic_num = atom.GetAtomicNum(),
                    degree = atom.GetTotalDegree(),
                    valence = atom.GetTotalValence(),
                    formal_charge = atom.GetFormalCharge(),
                    #hybridization = atom.GetHybridization(),
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
                    # mol features
                    num_h_acceptors = Lipinski.NumHAcceptors(mol),
                    num_h_donors = Lipinski.NumHDonors(mol),
                    #molwt = Descriptors.MolWt(mol),
                    #logp = Crippen.MolLogP(mol),
                    #tpsa = rdMolDescriptors.CalcTPSA(mol),
                    #labute_asa = rdMolDescriptors.CalcLabuteASA(mol),
                    # the next two elements are later used to compute the labels but will of course
                    # not be used as features!
                    mol_id = int(mol_id),
                    is_som = (atom_idx in soms))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type = bond.GetBondTypeAsDouble(),
                   bond_is_in_ring = bond.IsInRing(),
                   bond_is_aromatic = bond.GetIsAromatic(),
                   bond_is_conjugated = bond.GetIsConjugated(),
                   bond_stereo = bond.GetStereo())
    return G

def compute_node_features_matrix(G):
    """Takes as input a NetworkX Graph object (which already contains the 
    features for each individual nodes) and extracts/returns its corresponding node features matrix.

    Args:
        G (NetworkX Graph object)

    Returns:
        _features (numpy array): a numpy array of dimension (number of nodes, number of node features)
    """

    num_nodes = len(G.nodes)

    for i in tqdm(range(num_nodes)):
        current_node = G.nodes[i]
        # atom features
        atomic_num = [current_node['atomic_num']]
        degree = [current_node['degree']]
        valence = [current_node['valence']]
        formal_charge = [current_node['formal_charge']]
        #hybridization = [str(current_node['hybridization'])]
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
        # mol features
        num_h_acceptors = [current_node['num_h_acceptors']]
        num_h_donors = [current_node['num_h_donors']]
        #molwt = [current_node['molwt']]
        #logp = [current_node['logp']]
        #tpsa = [current_node['tpsa']]
        #labute_asa = [current_node['labute_asa']]

        features_vector = atomic_num + degree + valence + formal_charge + \
            num_hs + is_in_ring_3 + is_in_ring_4 + is_in_ring_5 + is_in_ring_6 + \
                is_in_ring_7 + is_in_ring_8 + is_aromatic + vdw_radius + covalent_radius + \
                    num_h_acceptors + num_h_donors #+ molwt  + logp + tpsa + labute_asa

        if i == 0:
            features = np.zeros((num_nodes, len(features_vector)), dtype=float)
        features[i,:] = np.array(features_vector)

    # One hot encode features
    ohe = OneHotEncoder(sparse=False, dtype=float, handle_unknown='infrequent_if_exist')
    features = ohe.fit_transform(features)

    return features

# def compute_edge_features_matrix(G):
#     """Takes as input a NetworkX Graph object (which already contains the 
#     features for each individual edge) and extracts/returns its corresponding edge features matrix.

#     Args:
#         G (NetworkX Graph object)

#     Returns:
#         _features (numpy array): a numpy array of dimension (number of edges, number of edge features)
#     """

#     # get features dimension:
#     num_edges = len(G.edges)
#     num_features = 4  # Need to automate this later, but this will do for now.

#     # construct features matrix of shape (number of edges, number of features)
#     features = np.zeros((num_edges, num_features))

#     # write features to features matrix
#     for i, edge in tqdm(enumerate(G.edges)):
#         start, end = edge
#         bond_type = [G.get_edge_data(start, end)['bond_type']]
#         bond_is_aromatic = [G.get_edge_data(start, end)['bond_is_aromatic']]
#         bond_is_conjugated = [G.get_edge_data(start, end)['bond_is_conjugated']]
#         bond_stereo = [G.get_edge_data(start, end)['bond_stereo']]

#         edge_features_vector = bond_type + bond_is_aromatic + \
#             bond_is_conjugated + bond_stereo

#         features[i,:] = np.array(edge_features_vector)

#     return features


def process_data(data_directory, data_name):
    """Computes and saves the necessary data (graph, features, labels, graph_ids)
    to create a PyTorch Geometric custom dataset from an SDF file containing molecules.

    Args:
        data_directory (string): the folder where the SDF file is stored.
        dat_name (string): the SDF file.
        output_directory (string): the folder to which the correlation matrix gets written.
    """
    # Import data from sdf file
    df = PandasTools.LoadSDF(os.path.join(data_directory, data_name), removeHs=True)
    df['soms'] = df['soms'].map(ast.literal_eval)

    # Generate networkx graphs from mols and save them in a json file
    df["G"] = df.apply(lambda x: mol_to_nx(x.mol_id, x.ROMol, x.soms), axis=1)
    G = nx.disjoint_union_all(df["G"].to_list())
    with open(os.path.join(data_directory, 'graph.json'), 'w') as f:
            f.write(json.dumps(json_graph.node_link_data(G)))

    # Generate and save list of labels
    labels = []
    for i in range(len(G.nodes)):
        labels.append(int(G.nodes[i]['is_som']))
    labels = np.array(labels)
    np.save(os.path.join(data_directory, 'labels.npy'), labels)

    # Generate and save list of mol ids
    mol_ids = []
    for i in range(len(G.nodes)):
        mol_ids.append(G.nodes[i]['mol_id'])
    mol_ids = np.array(mol_ids)
    np.save(os.path.join(data_directory, 'mol_ids.npy'), mol_ids)

    # Compute node features matrix and save it to node_features.npy
    node_features = compute_node_features_matrix(G)
    np.save(os.path.join(data_directory, 'node_features.npy'), node_features)

    df = pd.DataFrame(node_features)
    corr_matrix = df.corr()
    plt.imshow(corr_matrix, cmap='binary')
    plt.savefig(os.path.join(data_directory, 'correlation_matrix.png'))

    # # Compute edge features matrix and save it to edge_features.npy
    # edge_features = compute_edge_features_matrix(G)
    # np.save(data_path+'edge_features.npy', edge_features)