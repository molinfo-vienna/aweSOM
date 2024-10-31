from __future__ import annotations

from multiprocessing import Pool
from typing import Any, List, Tuple

import networkx as nx
import numpy as np
import warnings
import pandas as pd
from rdkit.Chem import(
    Mol, 
    MolToSmiles,
    RemoveHs,
    #, rdMolDescriptors,
)
from rdkit.Chem.rdchem import Atom, Bond

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


def _one_hot_encode(value: Any, valid_set: list[Any]) -> List[float]:
    """One-hot encodes `value` based on `valid_set`, with the last element as default."""
    if value not in valid_set:
        value = valid_set[-1]
    return [float(value == element) for element in valid_set]


def generate_bond_features(bond: Bond) -> List[float]:
    """Generates bond (edge) features for a given bond."""
    return _one_hot_encode(str(bond.GetBondType()), BOND_TYPE_STR) + [
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
    ]


# def generate_mol_features(mol: Mol) -> List[float]:
#     """Generates molecular features for a given molecule."""
#     return [
#         rdMolDescriptors.CalcExactMolWt(mol),
#         rdMolDescriptors.CalcCrippenDescriptors(mol)[0],  # logP
#         rdMolDescriptors.CalcLabuteASA(mol),
#         rdMolDescriptors.CalcTPSA(mol),
#         rdMolDescriptors.CalcNumHBA(mol),
#         rdMolDescriptors.CalcNumHBD(mol),
#         rdMolDescriptors.CalcNumHeavyAtoms(mol),
#         rdMolDescriptors.CalcNumHeteroatoms(mol),
#         rdMolDescriptors.CalcFractionCSP3(mol),
#         rdMolDescriptors.CalcNumRings(mol),
#         rdMolDescriptors.CalcNumHeterocycles(mol),
#         rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
#         rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
#         rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
#         rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
#         rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
#         rdMolDescriptors.CalcNumRotatableBonds(mol),
#         rdMolDescriptors.CalcNumAmideBonds(mol),
#     ]


def generate_node_features(atom: Atom) -> List[float]:
    """Generates node (atom) features for a given atom."""
    return _one_hot_encode(atom.GetAtomicNum(), ELEM_LIST)


def mol_to_nx(mol_id: int, mol: Mol, soms: List[int]) -> nx.Graph:
    """Converts a molecule (Mol) to a NetworkX graph with node and edge attributes."""
    G = nx.Graph()

    # Add nodes (atoms) to graph
    # mol_features = generate_mol_features(mol)
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        G.add_node(
            atom_idx,
            node_features=generate_node_features(atom),
            # mol_features=mol_features,
            smiles=MolToSmiles(mol),
            is_som=(atom_idx in soms),
            # The next two elements are only used later to assign
            # the predictions to the rights atoms.
            # They are **not** used a features.
            mol_id=int(mol_id),
            atom_id=int(atom_idx),
        )

    # Add edges (bonds) to graph
    for bond in mol.GetBonds():
        G.add_edge(
            # The next two elements are only used to identify bonds.
            # They are **not** used a features.
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_features=generate_bond_features(bond),
        )
    return G


def process_data_chunk(df_chunk: pd.DataFrame) -> nx.Graph:
    """Processes a chunk of data to generate molecular subgraphs."""
    df_chunk["G"] = df_chunk.apply(lambda x: mol_to_nx(x.ID, x.ROMol, x.soms), axis=1)
    return nx.disjoint_union_all(df_chunk["G"].to_list())


def generate_preprocessed_data(df: pd.DataFrame, num_workers: int) -> nx.Graph:
    """Generates a preprocessed molecular graph using multiple workers."""
    with Pool(num_workers) as pool:
        chunks = np.array_split(df, num_workers)
        graphs = pool.map(process_data_chunk, chunks)
    return nx.disjoint_union_all(graphs)


def remove_implicit_Hs(row) -> Tuple[Mol, List[int]]:
    """Removes implicit hydrogens from a molecule and updates the SoM indices."""
    # Set the label property to whether the atom is a SoM or not
    for atom in row.ROMol.GetAtoms():
        atom_id = atom.GetIdx()
        if atom_id in set(row.soms):
            atom.SetIntProp("label", 1)
        else:
            atom.SetIntProp("label", 0)

    # Remove hydrogens
    mol = RemoveHs(row.ROMol)
        
    # Reset the SOM list to the new indices
    try:
        new_soms = [atom.GetIdx() for atom in row["ROMol"].GetAtoms() if atom.GetIntProp("label") == 1]
        return mol, new_soms
    except:
        molid = row["ID"]
        warnings.warn(f"SoM label issue on molecule {molid}")
        return mol, []
    