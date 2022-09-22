# Standard imports
import json
import os
import os.path as osp
from itertools import product
from tqdm import tqdm
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# RDKit
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# NetworkX
import networkx as nx
from networkx.readwrite import json_graph

# PyTorch / PYG
import torch
from torch_geometric.data import(
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops

# sklearn
from sklearn.preprocessing import OneHotEncoder

from som_dataset import SOM


"""Provide the primary functions."""


def mol_to_nx(mol):
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


def generate_labels(df_group):
    assert len(df_group) == 1
    soms = df_group.soms_new.iloc[0]
    mol = df_group.ROMol.iloc[0]
    df_group['atom'] = [range(mol.GetNumAtoms())]
    df_group = df_group.explode('atom')
    df_group['label'] = [(atom_idx in soms) for atom_idx in range(mol.GetNumAtoms())]
    return df_group

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, use_chirality = True, hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms
    permitted_list_of_atoms =  ['C', 'N', 'S', 'O', 'H', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass = [float(atom.GetMass())]
    vdw_radius = [float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))]
    covalent_radius = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())))]
    
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass + vdw_radius + covalent_radius
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

def compute_features_from_mols(mols):
    """
    Input:
        mols:       [mol_1, mol_2, ....] ... a list of mols
    
    Output:
        features:   a 2-dimensional NumPy array of shape (number of atoms, number of features)
                    encoding the features of all the atoms in the input list of mols.
    """

    first = True
    for mol in tqdm(mols):
        
        # get feature dimensions
        n_atoms = mol.GetNumAtoms()
        n_features = len(get_atom_features(mol.GetAtomWithIdx(0)))

        # construct node feature matrix X of shape (number of atoms, number of features) for the current mol
        X = np.zeros((n_atoms, n_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
        X = torch.tensor(X, dtype = torch.float)

        # append the features tensor "X" of the current mol to the global feature tensor "features"
        if first == True:
            features = X
        else:
            features = torch.cat((features, X), dim=0)
        first = False

    return features


if __name__ == "__main__":
    
    # Import data from sdf file
    df = PandasTools.LoadSDF('db_preprocessed.sdf', removeHs=True)
    df['soms_new'] = df['soms_new'].map(ast.literal_eval)

    # Generate networkx graphs from mols and save them in a json file
    df["G"] = df["ROMol"].apply(mol_to_nx)

    G = nx.disjoint_union_all(df["G"].to_list())
    with open('data/graph.json', 'w') as f:
            f.write(json.dumps(json_graph.node_link_data(G)))

    # Generate and save labels
    # 1) Create a new df which does not contain 1 mol per row anymore, but 1 atom per row
    df_labeled = df.groupby('mol_id', as_index=False).apply(generate_labels)
    df_labeled["label"] = df_labeled["label"].astype(int)
    # 2) One-hot-encode labels
    ohe = OneHotEncoder(sparse=False, dtype=np.compat.long)
    labels_ohe = ohe.fit_transform((df_labeled["label"].to_numpy()).reshape(-1,1))
    # 3) Save one-hot-encoded labels to .npy file
    np.save('data/labels.npy', labels_ohe)

    # Save mol ids (this will hep us know which graph corresponds to which molecule when generating the PYG dataset)
    df_labeled["mol_id"] = df_labeled["mol_id"].astype(int).to_numpy()
    np.save('data/mol_ids.npy', df_labeled["mol_id"].to_numpy())

    # Generate and save features
    mols_list = df.ROMol.tolist()
    features = compute_features_from_mols(mols_list)
    np.save('data/features.npy', features.numpy())

    # Create PyTorch Geometric Dataset
    dataset = SOM(root='.')
    print(f'Successfully generated SOM dataset!')

    # Print dataset info
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Training/Validation/Test Split
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset = dataset[int(len(dataset)*0.9):]

    print(f'Training set: {len(train_dataset)} graphs.')
    print(f'Validation set: {len(val_dataset)} graphs.')
    print(f'Test set: {len(test_dataset)} graphs.')

    #  Data Loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    