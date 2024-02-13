from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
import os

# import torch

from multiprocessing import Pool
from rdkit.Chem import Mol as RDKitMol
from rdkit.Chem.rdchem import Atom as RDKitAtom
from rdkit.Chem.rdchem import Bond as RDKitBond
from rdkit.Chem import rdMolDescriptors
from typing import Any, List, Tuple

import CDPL.Chem as Chem
import CDPL.MolProp as MolProp
import CDPL.ForceField as ForceField
from CDPL.Chem import BasicMolecule as CDPKitMol
from CDPL.Chem import Atom as CDPKitAtom
from CDPL.Chem import Bond as CDPKitBond


ELEM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53, "OTHER"]
TOTAL_DEGREE = [1, 2, 3, 4, "OTHER"]
FORMAL_CHARGE = [-1, 0, 1, "OTHER"]
HYBRIDIZATION_TYPE = ["SP", "SP2", "SP3", "OTHER"]
RING_SIZE = [0, 3, 4, 5, 6, 7, 8, "OTHER"]
TOTAL_VALENCE = [1, 2, 3, 4, 5, 6, "OTHER"]
NUM_H_NEIGHBORS = [0, 1, 2, 3, "OTHER"]

BOND_TYPE_STR = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"]
BOND_TYPE_INT = [1, 2, 3, "OTHER"]
BOND_STEREO_STR = [
    "STEREONONE",
    "STEREOANY",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
]

CLASS1 = [i for i in range(4)]  # 3 total
CLASS2 = [i for i in range(22)]  # 21 total
CLASS3 = [
    1,
    2,
    3,
    4,
    5,
    11,
    12,
    14,
    15,
    17,
    18,
    19,
    23,
    24,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    57,
    58,
    59,
    60,
    61,
    62,
    64,
    65,
    66,
    70,
    71,
    72,
    73,
    75,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    87,
    88,
    89,
    90,
    92,
    93,
    94,
    95,
    98,
    99,
    100,
    103,
    104,
    105,
    107,
    108,
    110,
    111,
    112,
    113,
    115,
    116,
    117,
    118,
    119,
    121,
    122,
    123,
    125,
    126,
    127,
    129,
    155,
]  # 93 total


"""
General Utilities
"""


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


def _get_one_hot_encoded_list(lst: list[int], allowable_set: list[int]) -> List[float]:
    """
    PRIVATE method generates a one-hot encoded list for a list of inputs lst.
    Args:
        lst (list): list of target values
        allowable_set (list): the allowed set
    Returns:
        (list): one-hot encoded list
    """
    return [1.0 if x in lst else 0.0 for x in allowable_set]


"""
CDPKit Utilities
"""


def _get_atom_degree(atom: Chem.Atom, mol: Chem.BasicMolecule) -> int:
    """
    PRIVATE method that takes an CDPKit atom as input and returns an int or str value ('OTHER')
    for the total degree of the atom.
    It includes hydrogens. MolProp.getHeavyBondCount(atom) reports degree without hydrogens.
    A string value is returned when its another element than in the TOTAL_DEGREE.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
    Returns:
        degree (int): degree
        degree (str): 'OTHER'
    """
    degree = MolProp.getBondCount(atom, mol)
    if degree not in TOTAL_DEGREE:
        degree = "OTHER"

    return degree


def _get_atom_valence(atom: Chem.Atom, mol: Chem.BasicMolecule) -> int:
    """
    PRIVATE method that takes an CDPKit atom as input and returns an int or str value ('OTHER')
    for the valence of the atom.
    A string value is returned when its another element than in the TOTAL_VALENCE.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
        mol (CDPKit Molecule): The molecule the atom is part of
    Returns:
        valence (int): degree
        valence (str): 'OTHER'
    """
    valence = MolProp.calcValence(atom, mol)
    if valence not in TOTAL_VALENCE:
        valence = "OTHER"

    return valence


def _get_atom_type(atom: Chem.Atom) -> int:
    """
    PRIVATE method that takes an CDPKit atom as input and returns an int or str value ('OTHER')
    for the atom type.
    A string value is returned when its another element than in the ELEM_LIST.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
    Returns:
        atomic_no (int): atomic_no
        atomic_no (str): 'OTHER'
    """
    atomic_no = Chem.getType(atom)

    if atomic_no not in ELEM_LIST:
        atomic_no = "OTHER"

    return atomic_no


def _get_file_length(dir: str) -> int:
    """
    Loads the file based on either SDF or SMILES format and returns the length.
    Args:
        dir (str): directory to file (incl filename)
    Returns:
        int: length of file
    """
    reader = _get_reader_by_file_extention(dir)
    return reader.getNumRecords()


def _get_formal_charge(atom: Chem.Atom) -> int:
    """
    PRIVATE method that takes an CDPKit atom as input and returns an int or str value ('OTHER')
    for the formal charge of the atom.
    A string value is returned when its another element than in the FORMAL_CHARGE.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
    Returns:
        form_chg (int): form_chg
        form_chg (str): 'OTHER'
    """
    form_chg = Chem.getFormalCharge(atom)

    if form_chg not in FORMAL_CHARGE:
        form_chg = "OTHER"

    return form_chg


def _get_hybridization_type(atom: Chem.Atom) -> int:
    """
    PRIVATE method that takes an CDPKit atom as input and returns a str value
    for the hybridization type of the atom as in HYBRIDIZATION_TYPE.
    Args:
        atom (CDPKit Atom): the atom to calculate the property
    Returns:
        form_chg (int): form_chg
        form_chg (str): 'OTHER'
    """
    hyb_state = Chem.getHybridizationState(atom)
    if hyb_state == Chem.HybridizationState.SP1:
        hyb_state = "SP"
    elif hyb_state == Chem.HybridizationState.SP2:
        hyb_state = "SP2"
    elif hyb_state == Chem.HybridizationState.SP3:
        hyb_state = "SP3"
    else:
        hyb_state = "OTHER"

    return hyb_state


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

    if name_and_ext[1] == "":
        raise IOError(
            "Error: could not determine molecule input file format (file extension missing)"
        )

    # get input handler for the format specified by the input file's extension
    ipt_handler = Chem.MoleculeIOManager.getInputHandlerByFileExtension(
        name_and_ext[1][1:]
    )

    if not ipt_handler:
        raise IOError(
            "Error: unsupported molecule input file format '%s'" % name_and_ext[1]
        )
    # create and return file reader instance
    return ipt_handler.createReader(dir)


def _is_conjugated(bond: Chem.Bond, mol: Chem.BasicMolecule) -> bool:
    """
    PRIVATE method that takes a Bond and returns a flag for conjugated bonds
    Args:
        bond (CDPKit Bond): the bond to calculate the property
        mol (CDPKit Molecule): The molecule the bond is part of
    Returns:
        (Boolean): if conjugated (True) or not (False)
    """
    elec_sys_list = Chem.perceivePiElectronSystems(mol)
    is_conj = False
    for elec_sys in elec_sys_list:
        if elec_sys.getNumAtoms() < 3:
            continue

        if elec_sys.containsAtom(bond.getBegin()) and elec_sys.containsAtom(
            bond.getEnd()
        ):
            is_conj = True
            break
    return is_conj


def generate_bond_features_CDPKit(bond: CDPKitBond, mol: CDPKitMol) -> list[float]:
    """
    Generates the edge features for each bond.
    Args:
        bond (CDPKit Bond): bond for the features calculation
        mol (CDPKit Molecule): the molecule the bonds are part of
    Returns:
        (list[float]): one-hot encoded atom feature list
    """
    return _get_one_hot_encoded_element(Chem.getOrder(bond), BOND_TYPE_INT) + [
        float(Chem.getRingFlag(bond)),
        float(_is_conjugated(bond, mol)),
        float(Chem.getAromaticityFlag(bond)),
    ]


def generate_preprocessed_data_CDPKit(
    dir: str, file_length: int, num_workers: int, labels: bool
) -> Tuple[
    nx.Graph,
    np.ndarray[np.int64, Any],
    np.ndarray[np.int64, Any],
    np.ndarray[np.int64, Any],
    np.ndarray[np.float64, Any],
]:
    """
    Generates the necessary preprocessed data from the input data using multiple workers.
    Args:
        dir (str):          dir to file (incl filename)
        file_length (int):  length of the instances in the file
        numWorkers (int):   number of worker processes to use
        labels (bool):      whether or not the input data contains true labels

    Returns:
        G (NetworkX Graph): a molecular graph describing the entire input data
                            (individual molecules are processed as subgraphs of
                            one big graph object)
    """
    with Pool(num_workers) as p:
        results = p.starmap(
            load_mol_input,
            zip(
                [dir for _ in range(num_workers)],
                [labels for _ in range(num_workers)],
                np.array_split(range(file_length), num_workers),
            ),
        )

    G = nx.disjoint_union_all([graph for result in results for graph in result])

    return G


def generate_node_features_CDPKit(
    atom: CDPKitAtom,
    mol: CDPKitMol,
) -> List[float]:
    """
    Generates the node features for each atom
    Args:
        atom (CDPKit Atom): atom for the features calculation
        atm_ring_length (int)
        mol (CDPKit Molecule): molecule that is host of the atoms and used for the atom features calculations.
    Returns:
        (list[float]): one-hot encoded atom feature list
    """
    # features = {
    #     "atom_type": _get_one_hot_encoded_element(_get_atom_type(atom), ELEM_LIST),
    #     "formal_charge": _get_one_hot_encoded_element(_get_formal_charge(atom), FORMAL_CHARGE),
    #     "hybridization_state": _get_one_hot_encoded_element(
    #         _get_hybridization_type(atom), HYBRIDIZATION_TYPE
    #     ),
    #     "ring_size": _get_one_hot_encoded_element(atm_ring_length, RING_SIZE),
    #     "aromaticity": list([float(Chem.getAromaticityFlag(atom))]),
    #     "degree": _get_one_hot_encoded_element(_get_atom_degree(atom, mol), TOTAL_DEGREE),
    #     "valence": _get_one_hot_encoded_element(_get_atom_valence(atom, mol), TOTAL_VALENCE),
    # }
    # return (
    #     features["atom_type"]
    #     + features["formal_charge"]
    #     + features["hybridization_state"]
    #     + features["ring_size"]
    #     + features["aromaticity"]
    #     + features["degree"]
    #     + features["valence"]
    # )

    features = {
        "SybylType": [Chem.getSybylType(atom)],
        "HeavyAtomCount": [MolProp.getHeavyAtomCount(atom, mol)],
        "ExplicitValence": [MolProp.calcExplicitValence(atom, mol)],
        "HybridPolarizability": [MolProp.getHybridPolarizability(atom, mol)],
        "VSEPRCoordinationGeometry": [MolProp.getVSEPRCoordinationGeometry(atom, mol)],
        "EffectivePolarizability": [MolProp.calcEffectivePolarizability(atom, mol)],
        "InductiveEffect": [MolProp.calcInductiveEffect(atom, mol)],
        "PEOESigmaCharge": [MolProp.getPEOESigmaCharge(atom)],
        "PEOESigmaElectronegativity": [MolProp.getPEOESigmaElectronegativity(atom)],
        "PiElectronegativity": [MolProp.calcPiElectronegativity(atom, mol)],
        "MMFF94Charge": [ForceField.getMMFF94Charge(atom)],
    }
    return (
        features["SybylType"]
        + features["HeavyAtomCount"]
        + features["ExplicitValence"]
        + features["HybridPolarizability"]
        + features["VSEPRCoordinationGeometry"]
        + features["EffectivePolarizability"]
        + features["InductiveEffect"]
        + features["PEOESigmaCharge"]
        + features["PEOESigmaElectronegativity"]
        + features["PiElectronegativity"]
        + features["MMFF94Charge"]
    )


def load_mol_input(dir: str, labels: bool, indices: List[int]) -> Tuple[List[nx.Graph]]:
    """
    loads the dataset based on either SDF or SMILES format.
    Args:
        dir (str):           dir to file (incl. filename)
        labels (bool):       whether or not the input data contains true labels
        indices (list[int]): list of indices for the entry selection
    Returns:

    """
    reader = _get_reader_by_file_extention(dir)
    graphs: List[nx.Graph] = list()

    for i in indices:
        try:
            mol = Chem.BasicMolecule()
            reader.read(int(i), mol)
            if not Chem.hasStructureData(mol):
                raise Warning(
                    f"Warning: no structure data available for molecule {Chem.getName(mol)}"
                )
            struct_data = Chem.getStructureData(mol)
            soms = list()
            for (
                entry
            ) in (
                struct_data
            ):  # iterate of structure data entries consisting of a header line and the actual data
                if labels:
                    if "som" in entry.header.lower():
                        if len(entry.data) > 2:
                            soms = [
                                int(s)
                                for s in entry.data.replace("[", "")
                                .replace("]", "")
                                .split(",")
                            ]
                        else:
                            raise Warning(
                                f"Warning: empty list of SoMs for molecule {Chem.getName(mol)}"
                            )
                else:
                    soms = []

            Chem.calcImplicitHydrogenCounts(
                mol, False
            )  # calculate implicit hydrogen counts and set corresponding property for all atoms
            Chem.perceiveHybridizationStates(
                mol, False
            )  # perceive atom hybridization states and set corresponding property for all atoms
            Chem.perceiveSSSR(
                mol, False
            )  # perceive smallest set of smallest rings and store as Chem.MolecularGraph property
            Chem.setRingFlags(
                mol, False
            )  # perceive cycles and set corresponding atom and bond properties
            Chem.setAromaticityFlags(
                mol, False
            )  # perceive aromaticity and set corresponding atom and bond properties
            Chem.perceiveSybylAtomTypes(
                mol, False
            )  # perceive Sybyl atom types and set corresponding property for all atoms
            Chem.calcTopologicalDistanceMatrix(
                mol, False
            )  # calculate topological distance matrix and store as Chem.MolecularGraph property (required for effective polarizability calculations)
            Chem.perceivePiElectronSystems(
                mol, False
            )  # perceive pi electron systems and store info as Chem.MolecularGraph property (required for MHMO calculations)
            MolProp.calcPEOEProperties(
                mol, False
            )  # calculate sigma charges and electronegativities using the PEOE method and store values as atom properties (prerequisite for MHMO calculations)
            MolProp.calcMHMOProperties(
                mol, False
            )  # calculate pi charges, electronegativities and other properties by a modified Hueckel MO method and store values as properties
            ForceField.perceiveMMFF94AromaticRings(
                mol, False
            )  # perceive aromatic rings according to the MMFF94 aroamticity model and store data as Chem.MolecularGraph property
            ForceField.assignMMFF94AtomTypes(
                mol, False, False
            )  # perceive MMFF94 atom types (tolerant mode) set corresponding property for all atoms
            ForceField.assignMMFF94BondTypeIndices(
                mol, False, False
            )  # perceive MMFF94 bond types (tolerant mode) set corresponding property for all bonds
            ForceField.calcMMFF94AtomCharges(
                mol, False, False
            )  # calculate MMFF94 atom charges (tolerant mode) set corresponding property for all atoms

            # Chem.makeHydrogenDeplete(mol)  # remove explicit hydrogens

            Chem.perceiveComponents(mol, True)

            G = mol_to_nx_CDPKit(Chem.getName(mol), mol, soms)
            graphs.append(G)

        except Exception as e:
            print(
                f"An error occurred while loading molecule {Chem.getName(mol)}. Exception: {e}"
            )
            continue
    return graphs


def mol_to_nx_CDPKit(mol_id: int, mol: CDPKitMol, soms: List[int]) -> nx.Graph:
    """
    This function takes an CDPKit Mol object as input and returns its corresponding
    NetworkX graph with node and edge attributes.
    Args:
        mol_id (int): the molecular ID of the parsed mol
        mol (CDPKit Mol): a CDPKit Mol object
        soms (list): a list of the indices of atoms that are SoMs (This is
                    of course only relevant for training data. If there is no info
                    about which atom is a SoM, then the list is simply empty.)
    Returns:
        NetworkX Graph with node and edge attributes
    """
    G = nx.Graph()
    # Assign each atom its atomic features and make it a node of G
    for atom in mol.getAtoms():
        atom_idx = atom.getIndex()
        G.add_node(
            atom_idx,  # node identifier
            node_features=generate_node_features_CDPKit(atom, mol),
            is_som=(atom_idx in soms),  # label
            # the next two elements are later used to assign the
            # predicted labels but are of course not used as features!
            mol_id=int(mol_id),
            atom_id=int(atom_idx),
            # coordinates = torch.tensor(Chem.get3DCoordinates(atom))
        )
    for bond in mol.getBonds():
        G.add_edge(
            # the next two elements are only used to identify bonds,
            # they are not used a features.
            mol.getAtomIndex(bond.getBegin()),
            mol.getAtomIndex(bond.getEnd()),
            bond_features=generate_bond_features_CDPKit(bond, mol),
        )
    return G


"""
RDKit Utilities
"""


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


def generate_preprocessed_data_RDKit(
    df: pd.DataFrame, num_workers: int
) -> Tuple[
    nx.Graph,
    np.ndarray[np.int64, Any],
    np.ndarray[np.int64, Any],
    np.ndarray[np.int64, Any],
    np.ndarray[np.float64, Any],
]:
    """
    Generates the a preprocessed graph from the input data using multiple workers.
    Args:
        df (pandas dataframe): input data
        numWorkers (int): number of worker processes to use
    Returns:
        G (NetworkX Graph): a molecular graph describing the entire input data
                            (individual molecules are processed as subgraphs of
                            one big graph object)
    """
    chunks = np.array_split(df, num_workers)
    with Pool(num_workers) as p:
        results = p.map(generate_preprocessed_data_chunk_RDKit, chunks)
    G = nx.disjoint_union_all([result for result in results])

    return G


def generate_preprocessed_data_chunk_RDKit(
    df_chunk: pd.DataFrame,
) -> Tuple[
    nx.Graph,
    np.ndarray[np.int64, Any],
    np.ndarray[np.int64, Any],
    np.ndarray[np.int64, Any],
    np.ndarray[np.float64, Any],
]:
    """
    Generates preprocessed data from a chunk of the input data.
    """
    df_chunk["G"] = df_chunk.apply(
        lambda x: mol_to_nx_RDKit(x.ID, x.ROMol, x.soms),
        axis=1
        # lambda x: mol_to_nx_RDKit(x.ID, x.ROMol, x.soms, x.class3), axis=1
    )
    G = nx.disjoint_union_all(df_chunk["G"].to_list())

    return G


def generate_mol_features_RDKit(mol: RDKitMol) -> List[float]:
    return [
        rdMolDescriptors.CalcExactMolWt(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
        rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcTPSA(mol),
    ]


def generate_node_features_RDKit(atom: RDKitAtom, atm_ring_length: int) -> List[float]:
    # def generate_node_features_RDKit(atom: RDKitAtom, atm_ring_length: int, class3: int) -> List[float]:
    """
    Generates the node features for each atom
    Args:
        atom (RDKit Atom): atom for the features calculation
        atm_ring_length (int): the size of the ring in which the atom is
        class3 (int): the reported reaction subclass
    Returns:
        (list[float]): one-hot encoded atom feature list
    """
    features = {
        "atom_type": _get_one_hot_encoded_element(atom.GetAtomicNum(), ELEM_LIST),
        # "aromaticity": list([float(atom.GetIsAromatic())]),
        "formal_charge": _get_one_hot_encoded_element(
            atom.GetFormalCharge(), FORMAL_CHARGE
        ),
        # "hybridization_state": _get_one_hot_encoded_element(
        #     str(atom.GetHybridization()), HYBRIDIZATION_TYPE
        # ),
        # "num_h_neighbors": _get_one_hot_encoded_element(atom.GetTotalNumHs(), NUM_H_NEIGHBORS),
        # "ring_size": _get_one_hot_encoded_element(atm_ring_length, RING_SIZE),
        # "total_degree": _get_one_hot_encoded_element(atom.GetTotalDegree(), TOTAL_DEGREE),
        # "valence": _get_one_hot_encoded_element(atom.GetTotalValence(), TOTAL_VALENCE),
        # "class3": _get_one_hot_encoded_element(class3, CLASS3),
    }
    return (
        features["atom_type"]
        # + features["aromaticity"]
        + features["formal_charge"]
        # + features["hybridization_state"]
        # + features["num_h_neighbors"]
        # + features["ring_size"]
        # + features["total_degree"]
        # + features["valence"]
        # + features["class3"]
    )

def mol_to_nx_RDKit(mol_id: int, mol: RDKitMol, soms: list[int]) -> nx.Graph:
    # def mol_to_nx_RDKit(mol_id: int, mol: RDKitMol, soms: list[int], class3: int) -> nx.Graph:
    """
    This function takes an RDKit Mol object as input and returns its corresponding
    NetworkX graph with node and edge attributes.
    Args:
        mol_id (int): the molecular ID of the parsed mol
        mol (RDKit Mol): an RDKit Mol object
        soms (list): a list of the indices of atoms that are SoMs (This is
                    of course only relevant for training data. If there is no info
                    about which atom is a SoM, then the list is simply empty.)
        class3 (int): the reported reaction subclass
    Returns:
        NetworkX Graph with node and edge attributes
    """
    G = nx.Graph()

    # Assign each atom its features and make it a node of G
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atm_ring_length = 0
        if atom.IsInRing():
            for ring in mol.GetRingInfo().AtomRings():
                if atom_idx in ring:
                    atm_ring_length = len(ring)
        G.add_node(
            atom_idx,  # node identifier
            node_features=generate_node_features_RDKit(atom, atm_ring_length),
            # node_features=generate_node_features_RDKit(atom, atm_ring_length, class3),
            mol_features = generate_mol_features_RDKit(mol),
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
