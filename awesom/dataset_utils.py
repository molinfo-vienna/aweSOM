from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd

from multiprocessing import Pool
from rdkit.Chem import Mol as RDKitMol
from rdkit.Chem.rdchem import Atom as RDKitAtom
from rdkit.Chem.rdchem import Bond as RDKitBond
from rdkit.Chem import MolToSmiles, rdMolDescriptors
from typing import Any, List


ELEM_LIST = [
    5,
    6,
    7,
    8,
    9,
    14,
    15,
    16,
    17,
    35,
    53,
    "OTHER",
]  # B, C, N, O, F, Si, P, S, Cl, Br, I

BOND_TYPE_STR = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"]


def _get_one_hot_encoded_element(x: Any, allowable_set: list[Any]) -> List[float]:
    """
    PRIVATE method generates a one-hot encoded list for x. If x not in allowable_set,
    the last value of the allowable_set is taken.
    Args:
        x (list): list of target values
        allowable_set (list): the allowed set
    Returns:
        (list): one-hot encoded list
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def generate_bond_features_RDKit(bond: RDKitBond) -> list[float]:
    """
    Generates the edge features for each bond
    Args:
        bond (RDKit Bond): bond for the features calculation
    Returns:
        (list[float]): one-hot encoded atom feature list
    """
    return _get_one_hot_encoded_element(str(bond.GetBondType()), BOND_TYPE_STR) + [
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
    ]


def generate_preprocessed_data_RDKit(df: pd.DataFrame, num_workers: int) -> nx.Graph:
    """
    Generates the a preprocessed graph from the input data using multiple workers.
    Args:
        df (pandas dataframe): input data
        numWorkers (int): number of worker processes to use
    Returns:
        G (NetworkX Graph): a molecular graph describing the input data
                            (individual molecules are processed as subgraphs of
                            one big graph object)
    """
    chunks = np.array_split(df, num_workers)
    with Pool(num_workers) as p:
        results = p.map(generate_preprocessed_data_chunk_RDKit, chunks)
    G = nx.disjoint_union_all([result for result in results])

    return G


def generate_preprocessed_data_chunk_RDKit(df_chunk: pd.DataFrame) -> nx.Graph:
    """
    Generates preprocessed data from a chunk of the input data.
    Args:
        df_chunk (pandas dataframe): chunk of the input data
    Returns:
        G (NetworkX Graph): a molecular graph describing the input data
                            (individual molecules are processed as subgraphs of
                            one big graph object)
    """
    df_chunk["G"] = df_chunk.apply(
        lambda x: mol_to_nx_RDKit(x.ID, x.ROMol, x.soms), axis=1
    )
    G = nx.disjoint_union_all(df_chunk["G"].to_list())

    return G


def generate_mol_features_RDKit(mol: RDKitMol) -> List[float]:
    """
    Generates the molecular features for the molecule
    Args:
        mol (RDKit Mol): molecule for the features calculation
    Returns:
        (list[float]): list of molecular features
    """
    return [
        rdMolDescriptors.CalcExactMolWt(mol),
        rdMolDescriptors.CalcCrippenDescriptors(mol)[0],  # logP
        rdMolDescriptors.CalcLabuteASA(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
        rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
    ]


def generate_node_features_RDKit(atom: RDKitAtom) -> List[float]:
    """
    Generates the node features for each atom
    Args:
        atom (RDKit Atom): atom for the features calculation
    Returns:
        (list[float]): one-hot encoded atom feature list
    """
    features = {
        "atom_type": _get_one_hot_encoded_element(atom.GetAtomicNum(), ELEM_LIST)
    }
    return features["atom_type"]


def mol_to_nx_RDKit(mol_id: int, mol: RDKitMol, soms: list[int]) -> nx.Graph:
    """
    Takes an RDKit Mol object as input and returns its corresponding
    NetworkX graph with node and edge attributes.
    Args:
        mol_id (int): the molecular identifier of the parsed mol
        mol (RDKit Mol): an RDKit Mol object
        soms (list): a list of the indices of atoms that are SoMs (This is
                     of course only relevant for the training and testing data.
                     If there is no info about which atom is a SoM, then the list is empty.)
    Returns:
        NetworkX Graph with node and edge attributes
    """
    G = nx.Graph()

    # Assign each atom its features and make it a node of G
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        G.add_node(
            atom_idx,  # node identifier
            node_features=generate_node_features_RDKit(atom),
            mol_features=generate_mol_features_RDKit(
                mol
            ),  # the molecular features are generated by default and are used, or not, depending on the model architecture
            smiles=MolToSmiles(mol),
            is_som=(atom_idx in soms),  # label
            # the next two elements are later used to assign the
            # predicted labels but are of course not used as features!
            mol_id=int(mol_id),
            atom_id=int(atom_idx),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            # the next two elements are only used to identify bonds,
            # they are not used a features.
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_features=generate_bond_features_RDKit(bond),
        )
    return G
