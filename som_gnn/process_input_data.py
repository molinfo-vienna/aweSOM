import ast
import json
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, rdchem, rdMolDescriptors, PandasTools
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tqdm import tqdm

from rdkit.Chem import HybridizationType


#TODO I simply copy pasted here - Bond Stereo and others needs to be adjusted
HYBRIDIZATION_TYPE = [0,1,2,3,4,5,6,7,8],
TOTAL_DEGREE = [0,1,2,3,4,5,6,7,"unknown"],
ELEM_LIST =[1, 6, 7, 8, 16, 9, 15, 17, 35, 53,"unknown"],
FORMAL_CHARG = [-1,-2,1,2,"unknown"],
CIP_CONF = [8,2,4,"unknown"],
H_COUNT = [0,1,2,3,4,5,"unknown"],
TOTAL_VALENCE = [0,1,2,3,4,5,6,7,8,"unknown"],
RING_SIZE = [0,3,4,5,6,7,8,"other"],
H_COUNT = [0,1,2,3,4,5,"other"],
O_COUNT = [0,1,2,3,4,5,"other"],
N_COUNT = [0,1,2,3,4,5,"other"],
S_COUNT = [0,1,2,3,4,5,"other"],
HAL_COUNT = [0,1,2,3,4,5,"other"],
SP3C_COUNT = [0,1,2,3,4,5,"other"],
AR_COUNT = [0,1,2,3,4,5,"other"],
H_ACC_COUNT = [0,1,2,3,4,5,"other"],
H_DON_COUNT = [0,1,2,3,4,5,"other"],
RING_COUNT = [0,1,2,3,4,5,"other"]

ROT_BONDS_COUNT = [0,1,2,3,4,5,"other"]
BOND_STEREO = ["others"]
BOND_TYPE = [0,1,2,"others"]


def _getAllowedSet(x, allowable_set):
        '''  
        PRIVATE METHOD
        generates a one-hot encoded list for x. If x not in allowable_set,
        the last value of the allowable_set is taken. \n
        Input \n
            x (list): list of target values \n
            allowable_set (list): the allowed set \n
        Returns: \n
            (list): one-hot encoded list 
        '''
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))


def generateBondFeatures(bond):
    # bond_type=bond.GetBondTypeAsDouble(),
    # bond_is_in_ring=bond.IsInRing(),
    # bond_is_aromatic=bond.GetIsAromatic(),
    # bond_is_conjugated=bond.GetIsConjugated(),
    # bond_stereo=bond.GetStereo(),

    return ((_getAllowedSet(bond.GetBondTypeAsDouble(), BOND_TYPE)
                +_getAllowedSet(bond.GetStereo(), BOND_STEREO)
                +list([float(bond.IsInRing())])
                +list([float(bond.GetIsConjugated())])
                +list([float(bond.GetIsAromatic())])))

def generateNodeFeatures(atom,mol,atm_ring_length):
        ''' 
        generates the node features for each atom \n
        Input \n
        atom (RDKit Atom): atom for the features calculation \n
        mol (RDKit Molecule): molecule, the atom belongs to \n
        Return \n
        (list): one-hot encoded atom feature list
        '''
        # list(map(lambda s: float(x == s), allowable_set))
        # map(lambda atom.GetAtomicNum(),ELEM_LIST: _getAllowedSet)
        return ((_getAllowedSet(atom.GetAtomicNum(), ELEM_LIST)
                +_getAllowedSet(atom.GetTotalDegree(), TOTAL_DEGREE)
                +_getAllowedSet(atom.GetFormalCharge(), FORMAL_CHARG) 
                +_getAllowedSet(atom.GetHybridization(), HYBRIDIZATION_TYPE)
                +_getAllowedSet(atm_ring_length, RING_SIZE)
                +_getAllowedSet(atom.GetTotalNumHs(), H_COUNT))
                +_getAllowedSet(atom.GetTotalValence(), TOTAL_VALENCE)
                +_getAllowedSet(generate_num_element(mol, "O"), O_COUNT)
                +_getAllowedSet(generate_num_element(mol, "N"), N_COUNT)
                +_getAllowedSet(generate_num_element(mol, "S"), S_COUNT)
                +_getAllowedSet(generate_num_halogens(mol), HAL_COUNT)
                +_getAllowedSet(generate_num_sp3c(mol), SP3C_COUNT)
                +_getAllowedSet(rdMolDescriptors.CalcNumHBA(mol), H_ACC_COUNT)
                +_getAllowedSet(rdMolDescriptors.CalcNumHBD(mol), H_DON_COUNT)
                +_getAllowedSet(rdMolDescriptors.CalcNumRings(mol), RING_COUNT)
                +_getAllowedSet(rdMolDescriptors.CalcNumRotatableBonds(mol), ROT_BONDS_COUNT)
                +_getAllowedSet(len(mol.GetAromaticAtoms()), AR_COUNT)
                +list([float(atom.GetIsAromatic())])
                +list([float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))]) #TODO needs scaling
                +list([float(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()))]) #TODO needs scaling
                +list([float(rdMolDescriptors.CalcExactMolWt(mol))/700]) #TODO
                +list([float(np.log(rdMolDescriptors.CalcTPSA(mol)))])#TODO
                +list([float(np.log(Crippen.MolLogP(mol)))])#TODO
                +list([float(np.log(rdMolDescriptors.CalcCrippenDescriptors(mol)[1]))])#TODO
                +list([float(np.log(rdMolDescriptors.CalcLabuteASA(mol)))])#TODO
                +list([float(generate_fraction_rotatable_bonds(mol)*0.1)])#TODO
                )


def generate_fraction_rotatable_bonds(mol):
    """ Computes the fraction of rotatable bonds in the parsed molecule

    Args:
        mol (RDKit Mol): n RDKit Mol object

    Returns:
        float: the fraction of rotatable bond
    """
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_bonds = mol.GetNumBonds()

    if num_bonds == 0:
        return 0

    return num_rotatable_bonds / num_bonds


def generate_num_sp3c(mol):
    """Compute the number of SP3 hybridized carbon atoms
    in the parsed molecule

    Args:
        mol (RDKit Mol): an RDKit Mol object

    Returns:
        int: the number of SP3 hybridized carbon atoms
    """
    return len(
        [
            1
            for a in mol.GetAtoms()
            if a.GetHybridization() == rdchem.HybridizationType.SP3
        ]
    )


def generate_num_halogens(mol):
    """Computes the number of halogens in the parsed molecule

    Args:
        mol (RDKit Mol): an RDKit Mol object

    Returns:
        int: number of halogens
    """
    return len([1 for a in mol.GetAtoms() if a.GetSymbol() in ["F", "Cl", "Br", "I"]])


def generate_num_element(mol, element):
    """Computes the number of atoms corresponding to a specific element
    in the parsed molecule

    Args:
        mol (RDKit Mol): an RDKit Mol object
        element (string): the element to count

    Returns:
        int: number of atoms corresponding to parsed element
    """
    return len([1 for a in mol.GetAtoms() if a.GetSymbol() == element])


def mol_to_nx(mol_id, mol, soms):
    """Takes as input an RDKit mol object and return its corresponding NetworkX Graph

    Args:
        mol_id (int): the molecular ID of the parsed mol
        mol (RDKit Mol): an RDKit Mol object
        soms (list): a list of the indices of atoms that are SoMs
    Returns:
        NetworkX Graph with node and edge attributes
    """
    G = nx.Graph()
    # get ring info
    rings = mol.GetRingInfo().AtomRings()

    # Assign each atom its molecular and atomic features and make it a node of G
    # is_som = [(atom_idx in soms) for atom_idx in range(mol.GetNumAtoms())]
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atm_ring_length = 0
        if atom.IsInRing():
            for ring in rings:
                if atom_idx in ring:
                    atm_ring_length = len(ring)
        G.add_node(
            atom_idx,  # node identifier
            # Atomic Descriptors
            node_features=generateNodeFeatures(atom,mol,atm_ring_length),
            # the next two elements are later used to compute the labels but will
            # not be used as features!
            mol_id=int(mol_id),
            atom_id=int(atom_idx),
            is_som=(atom_idx in soms),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            # not used as features
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_features = generateBondFeatures(bond)
        )
    return G


def compute_node_features_matrix(G):
    """Takes as input a NetworkX Graph object (which already contains the
    features for each individual nodes) and extracts/returns its corresponding node features matrix.

    Args:
        G (NetworkX Graph object)

    Returns:
        features (numpy array): a numpy array of dimension (number of nodes, number of node features)
    """

    num_nodes = len(G.nodes)


    for i in tqdm(range(num_nodes)):
        current_node = G.nodes[i]
        node_features = current_node["node_features"]
        # bond_features = current_node["bond_features"]

    
    return np.array(node_features)
