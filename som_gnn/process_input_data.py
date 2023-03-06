import json
import networkx as nx
import numpy as np
import os
from rdkit.Chem import rdchem, rdMolDescriptors

HYBRIDIZATION_TYPE = ["UNSPECIFIED", 
                      "S", 
                      "SP", 
                      "SP2", 
                      "SP3", 
                      "SP2D", 
                      "SP3D", 
                      "SP3D2", 
                      "OTHER"]
TOTAL_DEGREE = [0,1,2,3,4,5,6,7,"OTHER"],
ELEM_LIST =[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53,"OTHER"]
FORMAL_CHARGE = [-1,-2,1,2,"OTHER"]
CIP_CONF = [8,2,4,"OTHER"]
H_COUNT = [0,1,2,3,4,5,"OTHER"]
TOTAL_VALENCE = [0,1,2,3,4,5,6,7,8,"OTHER"]
RING_SIZE = [0,3,4,5,6,7,8,"OTHER"]
H_COUNT = [0,1,2,3,4,5,6,7,8,9,"OTHER"]
O_COUNT = [0,1,2,3,4,5,"OTHER"]
N_COUNT = [0,1,2,3,4,5,"OTHER"]
S_COUNT = [0,1,2,3,4,5,"OTHER"]
HAL_COUNT = [0,1,2,3,4,5,"OTHER"],
SP3C_COUNT = [0,1,2,3,4,5,6,7,8,9,"OTHER"]
AR_COUNT = [0,1,2,3,4,5,6,7,8,9,"OTHER"]
H_ACC_COUNT = [0,1,2,3,4,5,6,7,8,9,"OTHER"]
H_DON_COUNT = [0,1,2,3,4,5,6,7,8,9,"OTHER"]
RING_COUNT = [0,1,2,3,4,5,"OTHER"]
ROT_BONDS_COUNT = [0,1,2,3,4,5,6,7,9,"OTHER"]

BOND_STEREO = ["STEREONONE", 
               "STEREOANY", 
               "STEREOZ", 
               "STEREOE", 
               "STEREOCIS", 
               "STEREOTRANS",
               "OTHER",
               ]
BOND_TYPE = ["UNSPECIFIED", 
             "SINGLE", 
             "DOUBLE", 
             "TRIPLE",
             "OTHER",
             ]


def _getAllowedSet(x, allowable_set):
        '''  
        PRIVATE METHOD
        Generates a one-hot encoded list for x. If x not in allowable_set,
        the last value of the allowable_set is taken.
        Args:
            x (list): list of target values
            allowable_set (list): the allowed set
        Returns:
            (list): one-hot encoded list 
        '''
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))


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


def compute_node_features_matrix(G):
    """Takes as input a NetworkX Graph object (which already contains the
    features for each individual nodes) and extracts/returns its corresponding node features matrix.

    Args:
        G (NetworkX Graph object)

    Returns:
        features (numpy array): a numpy array of dimension (number of nodes, number of node features)
    """

    num_nodes = len(G.nodes)
    node_features = np.empty((len(G.nodes()), len(G.nodes()[0]['node_features'])))

    for i in range(num_nodes):
        current_node = G.nodes[i]
        node_features[i,:] = (current_node["node_features"])

    return node_features


def generateBondFeatures(bond):

    return ((_getAllowedSet(bond.GetBondTypeAsDouble(), BOND_TYPE)
                +_getAllowedSet(bond.GetStereo(), BOND_STEREO)
                +list([float(bond.IsInRing())])
                +list([float(bond.GetIsConjugated())])
                +list([float(bond.GetIsAromatic())])))


def generateNodeFeatures(atom, mol, atm_ring_length):
        ''' 
        Generates the node features for each atom

        Args:
        atom (RDKit Atom): atom for the features calculation
        mol (RDKit Molecule): molecule, the atom belongs to

        Returns:
        (list): one-hot encoded atom feature list
        '''
        return ((_getAllowedSet(atom.GetAtomicNum(), ELEM_LIST)
                +_getAllowedSet(atom.GetTotalDegree(), TOTAL_DEGREE)
                +_getAllowedSet(atom.GetFormalCharge(), FORMAL_CHARGE) 
                +_getAllowedSet(atom.GetHybridization(), HYBRIDIZATION_TYPE)
                +_getAllowedSet(atm_ring_length, RING_SIZE)
                +_getAllowedSet(atom.GetTotalNumHs(), H_COUNT))
                +_getAllowedSet(atom.GetTotalValence(), TOTAL_VALENCE)
                +list([float(atom.GetIsAromatic())])
                # +_getAllowedSet(generate_num_element(mol, "O"), O_COUNT)
                # +_getAllowedSet(generate_num_element(mol, "N"), N_COUNT)
                # +_getAllowedSet(generate_num_element(mol, "S"), S_COUNT)
                # +_getAllowedSet(generate_num_halogens(mol), HAL_COUNT)
                # +_getAllowedSet(generate_num_sp3c(mol), SP3C_COUNT)
                # +_getAllowedSet(rdMolDescriptors.CalcNumHBA(mol), H_ACC_COUNT)
                # +_getAllowedSet(rdMolDescriptors.CalcNumHBD(mol), H_DON_COUNT)
                # +_getAllowedSet(rdMolDescriptors.CalcNumRings(mol), RING_COUNT)
                # +_getAllowedSet(rdMolDescriptors.CalcNumRotatableBonds(mol), ROT_BONDS_COUNT)
                # +_getAllowedSet(len(mol.GetAromaticAtoms()), AR_COUNT)
                # +list([float(generate_fraction_rotatable_bonds(mol))])
                )


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
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atm_ring_length = 0
        if atom.IsInRing():
            for ring in rings:
                if atom_idx in ring:
                    atm_ring_length = len(ring)
        G.add_node(
            atom_idx, # node identifier
            node_features=generateNodeFeatures(atom, mol, atm_ring_length),
            is_som=(atom_idx in soms), # label
            # the next two elements are later used to compute the labels but will
            # not be used as features!
            mol_id=int(mol_id),
            atom_id=int(atom_idx),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            # not used as features
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_features = generateBondFeatures(bond)
        )
    return G


def generate_preprocessed_data(df):

    # Generate networkx graphs from mols
    df["G"] = df.apply(lambda x: mol_to_nx(x.ID, x.ROMol, x.soms), axis=1)
    G = nx.disjoint_union_all(df["G"].to_list())

    # Compute list of mol ids
    mol_ids = []
    for i in range(len(G.nodes)):
        mol_ids.append(G.nodes[i]["mol_id"])
    mol_ids = np.array(mol_ids)

    # Compute list of atom ids
    atom_ids = []
    for i in range(len(G.nodes)):
        atom_ids.append(G.nodes[i]["atom_id"])
    atom_ids = np.array(atom_ids)

    # Compute list of labels
    labels = []
    for i in range(len(G.nodes)):
        labels.append(int(G.nodes[i]["is_som"]))
    labels = np.array(labels)
    
    # Compute node features matrix
    node_features = compute_node_features_matrix(G)

    return G, mol_ids, atom_ids, labels, node_features


def save_preprocessed_data(G, mol_ids, atom_ids, labels, node_features, dir):
    with open(os.path.join(dir, "graph.json"), "w") as f:
        f.write(json.dumps(nx.readwrite.json_graph.node_link_data(G)))
    np.save(os.path.join(dir, "mol_ids.npy"), mol_ids)
    np.save(os.path.join(dir, "atom_ids.npy"), atom_ids)
    np.save(os.path.join(dir, "labels.npy"), labels)
    np.save(os.path.join(dir, "node_features.npy"), node_features)
