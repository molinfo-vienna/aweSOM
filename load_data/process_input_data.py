import ast
import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from rdkit.Chem import PandasTools
from sklearn.preprocessing import OneHotEncoder

from load_data.compute_features import features_from_mols


def mol_to_nx(mol):
    """Takes as input an RDKit mol object and return its corresponding NetworkX Graph.

    Args:
        mol (RDKit mol object)

    Returns:
        G (NetworkX.Graph)
    """
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def labels_generator(df_group):
    """

    Args:
        df_group (Pandas DataFrame GroupBy object)

    Returns:
        df_group (Pandas DataFrame GroupBy object)
    """
    assert len(df_group) == 1
    soms = df_group.soms_new.iloc[0]
    mol = df_group.ROMol.iloc[0]
    df_group['atom'] = [range(mol.GetNumAtoms())]
    df_group = df_group.explode('atom')
    df_group['label'] = [(atom_idx in soms) for atom_idx in range(mol.GetNumAtoms())]
    return df_group


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
    df["G"] = df["ROMol"].apply(mol_to_nx)

    G = nx.disjoint_union_all(df["G"].to_list())
    with open('data/graph.json', 'w') as f:
            f.write(json.dumps(json_graph.node_link_data(G)))

    # Generate and save labels
    # 1)    Create a new df which does not contain 1 mol per row anymore, but 1 atom per row
    #       with the label True if the atom is a SoM and false otherwise.
    df_labeled = df.groupby('mol_id', as_index=False).apply(labels_generator)
    # 2)    One-hot-encode labels
    #ohe = OneHotEncoder(sparse=False, dtype=np.compat.long)
    #labels_ohe = ohe.fit_transform((df_labeled["label"].to_numpy()).reshape(-1,1))
    # 3)    Save labels to .npy file
    #np.save('data/labels.npy', labels_ohe)
    np.save('data/labels.npy', (df_labeled["label"].to_numpy()).reshape(-1,1).astype(int)[:,0])

    # Save mol ids (this will hep us know which graph corresponds to which molecule when generating the PYG dataset)
    df_labeled["mol_id"] = df_labeled["mol_id"].astype(int).to_numpy()
    np.save('data/mol_ids.npy', df_labeled["mol_id"].to_numpy())

    # Generate and save features
    mols_list = df.ROMol.tolist()
    features = features_from_mols(mols_list)
    np.save('data/features.npy', features.numpy())
