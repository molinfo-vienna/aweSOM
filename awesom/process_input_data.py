import json
import logging
import networkx as nx
import numpy as np
import os
import sys

from multiprocessing import Pool
from rdkit.Chem import rdMolDescriptors

import CDPL.Chem as Chem
import CDPL.MolProp as MolProp

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
BOND_TYPE = ["UNSPECIFIED", 
             "SINGLE", 
             "DOUBLE", 
             "TRIPLE",
             "OTHER",
             ]

def load_mol_input(dir: str, canonical_SMILES: bool) -> list(Chem.BasicMolecle):
    """
    loads the dataset based on either SDF or SMILES format.
    Args:
        dir (str): dir to file (incl filename)
        canonical_SMILES (boolean): Generate canonical SMILES
    Returns:
        list (CDPKit Mol)
        list (SMILES)
    """
    reader = _get_reader_by_file_extention(dir)

    smiles = list()
    mols = list()
    try:
        while True:
            mol = Chem.BasicMolecule()

            if not reader.read(mol):
                break
            
            Chem.calcImplicitHydrogenCounts(mol, False)  # calculate implicit hydrogen counts and set corresponding property for all atoms
            Chem.perceiveHybridizationStates(mol, False) # perceive atom hybridization states and set corresponding property for all atoms
            Chem.perceiveSSSR(mol, False)                # perceive smallest set of smallest rings and store as Chem.MolecularGraph property
            Chem.setRingFlags(mol, False)                # perceive cycles and set corresponding atom and bond properties
            Chem.setAromaticityFlags(mol, False)         # perceive aromaticity and set corresponding atom and bond properties
            Chem.makeHydrogenDeplete(mol)                # remove explicit hydrogens

            smile = Chem.generateSMILES(mol, canonical_SMILES)       # True -> canonical SMILES
            smiles.append(smile)

    except Exception as e: # handle exception raised in case of severe read errors
        sys.exit('Error: reading molecule failed: ' + str(e))

    return mols,smiles



def _get_reader_by_file_extention(dir: str) -> Chem.MoleculeReader:
    """
    PRIVATE METHOD
    Get input handler for the format specified by the input file's extension
    Args:
        dir (str): dir to file including name
    Returns:
        object (CDPKit MoleculeReader)
    """
    name_and_ext = os.path.splitext(dir)

    if name_and_ext[1] == '':
        sys.exit('Error: could not determine molecule input file format (file extension missing)')

    # get input handler for the format specified by the input file's extension
    ipt_handler = Chem.MoleculeIOManager.getInputHandlerByFileExtension(name_and_ext[1][1:])

    if not ipt_handler:
        sys.exit('Error: unupported molecule input file format \'%s\'' % name_and_ext[1])
    # create and return file reader instance
    return ipt_handler.createReader(dir)


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


def generate_fraction_rotatable_bonds_RDKit(mol):
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

def generate_fraction_rotatable_bonds(mol):
    """
    Computes the fraction of rotatable bonds in the parsed molecule.
    Args:
        mol (CDPKit Mol): CDPKit Mol object
    Returns:
        float: the fraction of rotatable bond
    """
    num_rotatable_bonds = MolProp.getRotatableBondCount(mol)
    num_bonds = mol.numAtoms()

    if num_bonds == 0:
        return 0

    return num_rotatable_bonds / num_bonds

def generate_fraction_HBA_RDKit(mol):
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

def generate_fraction_HBA(mol): 
    """
    Computes the fraction of hydrogen bond acceptors in the parsed molecule.
    Args:
        mol (CDPKit Mol): CDPKit Mol object
    Returns:
        float: the fraction of rotatable bond
    """
    num_HBA = MolProp.getHBondAcceptorAtomCount(mol)
    num_atoms = mol.numAtoms()

    if num_atoms == 0:
        return 0

    return num_HBA / num_atoms

def generate_fraction_HBD_RDKit(mol):
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


def generate_fraction_HBD(mol): 
    """
    Computes the fraction of hydrogen bond donors in the parsed molecule.
    Args:
        mol (CDPKit Mol): CDPKit Mol object
    Returns:
        float: the fraction of rotatable bond
    """
    num_HBD = MolProp.getHBondDonorAtomCount(mol)
    num_atoms = mol.numAtoms()

    if num_atoms == 0:
        return 0

    return num_HBD / num_atoms

def generate_fraction_element_RDKit(mol, element):
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


def generate_fraction_element(mol, element): 
    """
    Computes the fraction of atoms corresponding to a specific element
    in the parsed molecule.
    Args:
        mol (CDPKit Mol): CDPKit Mol object
        element (string): the element to count
    Returns:
        int: the number of atoms corresponding to the parsed element
    """
    return len([1 for a in mol.atoms() if Chem.getType(a) == element]) / mol.numAtoms()


def generate_fraction_halogens(mol): 
    """
    Computes the fraction of halogens in the parsed molecule
    Args:
        mol (CDPKit Mol): CDPKit Mol object
    Returns:
        int: the number of halogens
    """
    return len([1 for a in mol.atoms() if Chem.getType(a) in ["F", "Cl", "Br", "I"]]) / mol.numAtoms()

def generate_fraction_halogens_RDKit(mol):
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
        mol (CDPKit Mol): CDPKit Mol object
    Returns:
        float: the fraction of rotatable bond
    """
    num_ar = len(MolProp.getAromaticAtomCount(mol))
    num_atoms = mol.numAtoms()

    if num_atoms == 0:
        return 0

    return num_ar / num_atoms

def generate_fraction_aromatics_RDKit(mol):
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


def generate_bond_features(bond,mol): 
    """
    Generates the edge features for each bond
    Args:
        bond (CDPKit Bond): bond for the features calculation
        mol (CDPKit Molecule): The molecule the bonds are part of
    Returns:
        (list): one-hot encoded atom feature list
    """
    return (_get_allowed_set(float(Chem.getOrder(bond)), BOND_TYPE) #TODO bond.GetBondTypeAsDouble() does not return BOND_TYPE!!!
            + _get_allowed_set(Chem.calcBondStereoDescriptors(mol, False), BOND_STEREO)
            + [float(Chem.getRingFlag(bond)), 
               float(_is_conjugated(bond,mol)), 
               float(Chem.getAromaticityFlag(bond))
              ]
           )

def generate_bond_features_RDKit(bond):
    """
    Generates the edge features for each bond
    Args:
        bond (RDKit Bond): bond for the features calculation
    Returns:
        (list): one-hot encoded atom feature list
    """
    return (_get_allowed_set(bond.GetBondTypeAsDouble(), BOND_TYPE)
            + _get_allowed_set(bond.GetStereo(), BOND_STEREO)
            + [float(bond.IsInRing()), 
               float(bond.GetIsConjugated()), 
               float(bond.GetIsAromatic())
              ]
           )

def generate_node_features(atom, atm_ring_length, molecular_features,mol): 
    """
    Generates the node features for each atom
    Args:
        atom (CDPKit Atom): atom for the features calculation
        atm_ring_length (int)
        molecular_features (list of floats): a precomputed list of
            molecular features for the mol to which to function is
            applied to
        mol (CDPKit Molecule): molecule that is host of the atoms and used for the atom features calculations.
    Returns:
        (list): one-hot encoded atom feature list
    """
    features = {"atom_type": _get_allowed_set(_get_atom_type(atom), ELEM_LIST),
                "formal_charge": _get_allowed_set(_get_formal_charge(atom), FORMAL_CHARGE), 
                "hybridization_state": _get_allowed_set(_get_hybridization_type(atom), HYBRIDIZATION_TYPE), 
                "ring_size": _get_allowed_set(atm_ring_length, RING_SIZE), 
                "aromaticity": list([float(Chem.getAromaticityFlag(atom))]), 
                "degree": _get_allowed_set(_get_atom_degree(atom), TOTAL_DEGREE), 
                "valence": _get_allowed_set(_get_atom_valence(atom,mol), TOTAL_VALENCE), 
                "molecular": molecular_features
    }
    return (features["atom_type"]
            +features["formal_charge"]
            +features["hybridization_state"]
            +features["ring_size"]
            +features["aromaticity"]
            +features["degree"]
            +features["valence"])


def generate_node_features_RDKit(atom, atm_ring_length, molecular_features):
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

def _get_hybridization_type(atom: Chem.Atom) -> int:
    '''
    PRIVATE method that takes an CDPKit atom as input and returns a str value
    for the hybridization type of the atom as in HYBRIDIZATION_TYPE.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
    Returns:
        form_chg (int): form_chg
        form_chg (str): 'OTHER'
    '''
    hyb_state = Chem.getHybridizationState(atom)

    if hyb_state == Chem.HybridizationState.SP1: 
        hyb_state = 'SP'
    elif hyb_state == Chem.HybridizationState.SP2:
        hyb_state = 'SP2'
    elif hyb_state == Chem.HybridizationState.SP3:
        hyb_state = 'SP3'
    else:
        hyb_state = 'OTHER'
    
    return hyb_state

def _get_atom_valence(atom: Chem.Atom, mol: Chem.BasicMolecule) -> int:
    '''
    PRIVATE method that takes an CDPKit atom as input and returns an int or str value ('OTHER')
    for the valence of the atom.
    A string value is returned when its another element than in the TOTAL_VALENCE.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
        mol (CDPKit Molecule): The molecule the atom is part of
    Returns:
        valence (int): degree
        valence (str): 'OTHER'
    '''
    valence = MolProp.calcValence(atom, mol) # TS

    if valence not in TOTAL_VALENCE:         # TS
        valence = 'OTHER'
    
    return valence

def _get_atom_degree(atom: Chem.Atom) -> int:
    '''
    PRIVATE method that takes an CDPKit atom as input and returns an int or str value ('OTHER')
    for the total degree of the atom.
    It includes hydrogens. MolProp.getHeavyBondCount(atom) reports degree without hydrogens.
    A string value is returned when its another element than in the TOTAL_DEGREE.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
    Returns:
        degree (int): degree
        degree (str): 'OTHER'
    '''
    degree = MolProp.getBondCount(atom) # TS: degree including Hs! MolProp.getHeavyBondCount(atom) reports degree without Hs

    if degree not in TOTAL_DEGREE:
        degree = 'OTHER'
    
    return degree

def _get_formal_charge(atom: Chem.Atom) -> int:
    '''
    PRIVATE method that takes an CDPKit atom as input and returns an int or str value ('OTHER')
    for the formal charge of the atom. 
    A string value is returned when its another element than in the FORMAL_CHARGE.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
    Returns:
        form_chg (int): form_chg
        form_chg (str): 'OTHER'
    '''
    form_chg = Chem.getFormalCharge(atom)

    if form_chg not in FORMAL_CHARGE:
        form_chg = 'OTHER'

    return form_chg 

def _get_atom_type(atom: Chem.Atom) -> int:
    '''
    PRIVATE method that takes an CDPKit atom as input and returns an int or str value ('OTHER')
    for the atom type. 
    A string value is returned when its another element than in the ELEM_LIST.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
    Returns:
        atomic_no (int): atomic_no
        atomic_no (str): 'OTHER'
    '''
    atomic_no = Chem.getType(atom)

    if atomic_no not in ELEM_LIST:
        atomic_no = 'OTHER'

    return atomic_no 

def _is_conjugated(bond: Chem.Bond, mol: Chem.BasicMolecule) -> bool:
    '''
    PRIVATE method that takes a Bond and returns a flag for conjugated bonds
    Args:
        bond (CDPKit Bond): the bond to calculate the property
        mol (CDPKit Molecule): The molecule the bond is part of
    Returns:
        (Boolean): if conjugated (True) or not (False)
    '''
    elec_sys_list = Chem.perceivePiElectronSystems(mol)
    is_conj = False                  
        
    for elec_sys in elec_sys_list:  
        if elec_sys.numAtoms < 3:    
            continue                 

        if elec_sys.containsAtom(bond.getBegin()) and elec_sys.containsAtom(bond.getEnd()): 
            is_conj = True
            break
    return is_conj

def mol_to_nx(mol_id, mol, soms):
    """
    This function takes an CDPKit Mol object as input and returns its corresponding 
    NetworkX graph with node and edge attributes.
    Args:
        mol_id (int): the molecular ID of the parsed mol
        mol (CDPKit Mol): an CDPKit Mol object
        soms (list): a list of the indices of atoms that are SoMs (This is 
                    of course only relevant for training data. If there is no info
                    about which atom is a SoM, then the list is simply empty.)
    Returns:
        NetworkX Graph with node and edge attributes
    """
    G = nx.Graph()

    # Get ring info
    #rings = mol.GetRingInfo().AtomRings()
    rings = Chem.getSSSR(mol)

    # Assign each atom its molecular and atomic features and make it a node of G
    for atom_idx in range(len(mol.atoms())):
        atom = mol.GetAtomWithIdx(atom_idx)
        atm_ring_length = 0
        if atom.IsInRing():
            for ring in rings:
                if atom_idx in ring:
                    atm_ring_length = ring.numAtoms 
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
            node_features=generate_node_features(atom, atm_ring_length, molecular_features,mol),
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

def mol_to_nx_RDKit(mol_id, mol, soms):
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
        molecular_features = [float(generate_fraction_element_RDKit(mol, "N")),
                              float(generate_fraction_element_RDKit(mol, "O")),
                              float(generate_fraction_element_RDKit(mol, "S")),
                              float(generate_fraction_halogens_RDKit(mol)),
                              float(generate_fraction_rotatable_bonds_RDKit(mol)),
                              float(generate_fraction_HBA_RDKit(mol)),
                              float(generate_fraction_HBD_RDKit(mol)),
                              float(generate_fraction_aromatics_RDKit(mol))
                             ] + _get_allowed_set(rdMolDescriptors.CalcNumRings(mol), RING_COUNT)
        G.add_node(
            atom_idx, # node identifier
            node_features=generate_node_features_RDKit(atom, atm_ring_length, molecular_features),
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
            bond_features = generate_bond_features_RDKit(bond)
        )
    return G

def generate_preprocessed_data_chunk(df_chunks):
    """
    Generates preprocessed data from a chunk of the input data.
    """
    # Generate networkx graphs from mols
    df_chunk = df_chunks[0]

    if df_chunks[1]:
        df_chunk["G"] = df_chunk.apply(lambda x: mol_to_nx_RDKit(x.ID, x.mol, x.soms), axis=1)
    else:
        df_chunk["G"] = df_chunk.apply(lambda x: mol_to_nx(x.ID, x.mol, x.soms), axis=1)
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


def generate_preprocessed_data(df, num_workers, RD):
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
        RD (bool): If the calculations should be done via RDKit or CDPKit (default)
    """
    chunks = np.array_split(df, num_workers)
    with Pool(num_workers) as p:
        results = p.map(generate_preprocessed_data_chunk, (chunks,RD))
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
