import json
import logging
import networkx as nx
import numpy as np
import os
import sys

from multiprocessing import Pool
from rdkit.Chem import rdMolDescriptors

ELEM_LIST =[5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53,"OTHER"]
HYBRIDIZATION_TYPE = ["SP", 
                      "SP2", 
                      "SP3", 
                      "OTHER"]
FORMAL_CHARGE = [-1,0,1,"OTHER"]
RING_SIZE = [0,3,4,5,6,7,8,"OTHER"]
TOTAL_DEGREE = [1,2,3,4,"OTHER"]
TOTAL_VALENCE = [1,2,3,4,5,6,"OTHER"]
RING_COUNT = [0,1,2,3,4,5,"OTHER"]

BOND_STEREO = ["STEREONONE", 
               "STEREOANY", 
               "STEREOZ", 
               "STEREOE", 
               "STEREOCIS", 
               "STEREOTRANS",
               "OTHER",
               ]
BOND_TYPE = ["SINGLE", 
             "DOUBLE", 
             "TRIPLE",
             "OTHER",
             ]


def _get_allowed_set(x, allowable_set):
    """
    PRIVATE METHOD
    Generates a one-hot encoded list for x. If x not in allowable_set,
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


def generate_fraction_rotatable_bonds(mol):
    """
    Computes the fraction of rotatable bonds in the parsed molecule.
    Args:
        mol (RDKit Mol): RDKit Mol object
    Returns:
        float: the fraction of rotatable bond
    """
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_bonds = mol.GetNumBonds()

    if num_bonds == 0:
        return 0

    return num_rotatable_bonds / num_bonds


def generate_fraction_HBA(mol):
    """
    Computes the fraction of hydrogen bond acceptors in the parsed molecule.
    Args:
        mol (RDKit Mol): RDKit Mol object
    Returns:
        float: the fraction of rotatable bond
    """
    num_HBA = rdMolDescriptors.CalcNumHBA(mol)
    num_atoms = mol.GetNumAtoms()

    if num_atoms == 0:
        return 0

    return num_HBA / num_atoms


def generate_fraction_HBD(mol):
    """
    Computes the fraction of hydrogen bond donors in the parsed molecule.
    Args:
        mol (RDKit Mol): RDKit Mol object
    Returns:
        float: the fraction of rotatable bond
    """
    num_HBD = rdMolDescriptors.CalcNumHBD(mol)
    num_atoms = mol.GetNumAtoms()

    if num_atoms == 0:
        return 0

    return num_HBD / num_atoms


def generate_fraction_element(mol, element):
    """
    Computes the fraction of atoms corresponding to a specific element
    in the parsed molecule.
    Args:
        mol (RDKit Mol): RDKit Mol object
        element (string): the element to count
    Returns:
        int: the number of atoms corresponding to the parsed element
    """
    return len([1 for a in mol.GetAtoms() if a.GetSymbol() == element]) / mol.GetNumAtoms()


def generate_fraction_halogens(mol):
    """
    Computes the fraction of halogens in the parsed molecule
    Args:
        mol (RDKit Mol): RDKit Mol object
    Returns:
        int: the number of halogens
    """
    return len([1 for a in mol.GetAtoms() if a.GetSymbol() in ["F", "Cl", "Br", "I"]]) / mol.GetNumAtoms()


def generate_fraction_aromatics(mol):
    """
    Computes the fraction of aromatic atoms in the parsed molecule.
    Args:
        mol (RDKit Mol): RDKit Mol object
    Returns:
        float: the fraction of rotatable bond
    """
    num_ar = len(mol.GetAromaticAtoms())
    num_atoms = mol.GetNumAtoms()

    if num_atoms == 0:
        return 0

    return num_ar / num_atoms


def compute_node_features_matrix(G):
    """
    Takes as input a NetworkX Graph object (which already contains the
    features for each individual nodes) and extracts/returns its 
    corresponding node features matrix.
    Args:
        G (NetworkX Graph object)
    Returns:
        features (numpy array): a 2D array of dimension (number of nodes, 
                                number of node features)
    """
    num_nodes = len(G.nodes)
    node_features = np.empty((len(G.nodes()), len(G.nodes()[0]['node_features'])))

    for i in range(num_nodes):
        current_node = G.nodes[i]
        node_features[i,:] = (current_node["node_features"])
    return node_features


def generate_bond_features(bond):
    """
    Generates the edge features for each bond
    Args:
        bond (RDKit Bond): bond for the features calculation
    Returns:
        (list): one-hot encoded atom feature list
    """
    return (_get_allowed_set(bond.GetBondType(), BOND_TYPE)
            + _get_allowed_set(bond.GetStereo(), BOND_STEREO)
            + [float(bond.IsInRing()), 
               float(bond.GetIsConjugated()), 
               float(bond.GetIsAromatic())
              ]
           )

def generate_node_features(atom, atm_ring_length, molecular_features):
    """
    Generates the node features for each atom
    Args:
        atom (RDKit Atom): atom for the features calculation
        atm_ring_length (int)
        molecular_features (list of floats): a precomputed list of
            molecular features for the mol to which to function is
            applied to
    Returns:
        (list): one-hot encoded atom feature list
    """
    features = {"atom_type": _get_allowed_set(atom.GetAtomicNum(), ELEM_LIST),
                "formal_charge": _get_allowed_set(atom.GetFormalCharge(), FORMAL_CHARGE), 
                "hybridization_state": _get_allowed_set(str(atom.GetHybridization()), HYBRIDIZATION_TYPE), 
                "ring_size": _get_allowed_set(atm_ring_length, RING_SIZE), 
                "aromaticity": list([float(atom.GetIsAromatic())]), 
                "degree": _get_allowed_set(atom.GetTotalDegree(), TOTAL_DEGREE), 
                "valence": _get_allowed_set(atom.GetTotalValence(), TOTAL_VALENCE), 
                "molecular": molecular_features
    }
    return (features["atom_type"]
            +features["formal_charge"]
            +features["hybridization_state"]
            +features["ring_size"]
            +features["aromaticity"]
            +features["degree"]
            +features["valence"])


def mol_to_nx(mol_id, mol, soms):
    """
    This function takes an RDKit Mol object as input and returns its corresponding 
    NetworkX graph with node and edge attributes.
    Args:
        mol_id (int): the molecular ID of the parsed mol
        mol (RDKit Mol): an RDKit Mol object
        soms (list): a list of the indices of atoms that are SoMs (This is 
                    of course only relevant for training data. If there is no info
                    about which atom is a SoM, then the list is simply empty.)
    Returns:
        NetworkX Graph with node and edge attributes
    """
    G = nx.Graph()

    # Get ring info
    rings = mol.GetRingInfo().AtomRings()

    # Assign each atom its molecular and atomic features and make it a node of G
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atm_ring_length = 0
        if atom.IsInRing():
            for ring in rings:
                if atom_idx in ring:
                    atm_ring_length = len(ring)
        # Compute molecular features here so that we don't do it again and again
        # for each atom in generate_node_features in case FC4 is called for.
        molecular_features = [float(generate_fraction_element(mol, "N")),
                              float(generate_fraction_element(mol, "O")),
                              float(generate_fraction_element(mol, "S")),
                              float(generate_fraction_halogens(mol)),
                              float(generate_fraction_rotatable_bonds(mol)),
                              float(generate_fraction_HBA(mol)),
                              float(generate_fraction_HBD(mol)),
                              float(generate_fraction_aromatics(mol))
                             ] + _get_allowed_set(rdMolDescriptors.CalcNumRings(mol), RING_COUNT)
        G.add_node(
            atom_idx, # node identifier
            node_features=generate_node_features(atom, atm_ring_length, molecular_features),
            is_som=(atom_idx in soms), # label
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
            bond_features = generate_bond_features(bond)
        )
    return G


def generate_preprocessed_data_chunk(df_chunk):
    """
    Generates preprocessed data from a chunk of the input data.
    """
    # Generate networkx graphs from mols
    df_chunk["G"] = df_chunk.apply(lambda x: mol_to_nx(x.ID, x.ROMol, x.soms), axis=1)
    G = nx.disjoint_union_all(df_chunk["G"].to_list())
    # Compute list of mol ids
    mol_ids = np.array([G.nodes[i]["mol_id"] for i in range(len(G.nodes))])
    # Compute list of atom ids
    atom_ids = np.array([G.nodes[i]["atom_id"] for i in range(len(G.nodes))])
    # Compute list of labels
    labels = np.array([int(G.nodes[i]["is_som"]) for i in range(len(G.nodes))])
    # Compute node features matrix
    node_features = compute_node_features_matrix(G)

    return G, mol_ids, atom_ids, labels, node_features


def generate_preprocessed_data(df, num_workers):
    """
    Generates the necessary preprocessed data from the input data using multiple workers.
    Args:
        df (pandas dataframe): input data
        numWorkers (int): number of worker processes to use
    Returns:
        G (NetworkX Graph): a molecular graph describing the entire input data
                            (individual molecules are processed as subgraphs of 
                            one big graph object)
        mol_ids (numpy array): an array with the molecular ID of each node in G
                                (i.e. the molecule, to which each atom belongs to)
        atom_ids (numpy array): an array with the atom ID of each node in G
        labels (numpy array): an array with the SoM labels (0/1) of each node in G
        node_features (numpy array): a 2D array of dimension (number of nodes, 
                        number of node features)
    """
    chunks = np.array_split(df, num_workers)
    with Pool(num_workers) as p:
        results = p.map(generate_preprocessed_data_chunk, chunks)
    G = nx.disjoint_union_all([result[0] for result in results])
    mol_ids = np.concatenate([result[1] for result in results])
    atom_ids = np.concatenate([result[2] for result in results])
    labels = np.concatenate([result[3] for result in results])
    node_features = np.concatenate([result[4] for result in results], axis=0)

    return G, mol_ids, atom_ids, labels, node_features


def save_preprocessed_data(G, mol_ids, atom_ids, labels, node_features, dir):
    with open(os.path.join(dir, "graph.json"), "w") as f:
        f.write(json.dumps(nx.readwrite.json_graph.node_link_data(G)))
    np.save(os.path.join(dir, "mol_ids.npy"), mol_ids)
    np.save(os.path.join(dir, "atom_ids.npy"), atom_ids)
    np.save(os.path.join(dir, "labels.npy"), labels)
    np.save(os.path.join(dir, "node_features.npy"), node_features)
