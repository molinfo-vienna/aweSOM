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

    # Assign each atom its molecular and atomic features and make it a node of G
    # is_som = [(atom_idx in soms) for atom_idx in range(mol.GetNumAtoms())]
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        G.add_node(
            atom_idx,  # node identifier
            # Atomic Descriptors
            atomic_num=atom.GetAtomicNum(),
            degree=atom.GetTotalDegree(),
            valence=atom.GetTotalValence(),
            formal_charge=atom.GetFormalCharge(),
            hybridization=int(atom.GetHybridization()),
            num_hs=atom.GetTotalNumHs(),
            is_in_ring_3=atom.IsInRingSize(3),
            is_in_ring_4=atom.IsInRingSize(4),
            is_in_ring_5=atom.IsInRingSize(5),
            is_in_ring_6=atom.IsInRingSize(6),
            is_in_ring_7=atom.IsInRingSize(7),
            is_in_ring_8=atom.IsInRingSize(8),
            is_aromatic=atom.GetIsAromatic(),
            vdw_radius=Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()),
            covalent_radius=Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()),
            # Molecular Descriptors
            num_h_acceptors=rdMolDescriptors.CalcNumHBA(mol),
            num_h_donors=rdMolDescriptors.CalcNumHBD(mol),
            num_rings=rdMolDescriptors.CalcNumRings(mol),
            num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
            molwt=rdMolDescriptors.CalcExactMolWt(mol),
            tpsa=rdMolDescriptors.CalcTPSA(mol),
            logp=Crippen.MolLogP(mol),
            refractivity=rdMolDescriptors.CalcCrippenDescriptors(mol)[1],
            labute_asa=rdMolDescriptors.CalcLabuteASA(mol),
            frac_rotatable_bonds=generate_fraction_rotatable_bonds(mol),
            num_O=generate_num_element(mol, "O"),
            num_N=generate_num_element(mol, "N"),
            num_S=generate_num_element(mol, "S"),
            num_hal=generate_num_halogens(mol),
            num_sp3C=generate_num_sp3c(mol),
            num_ar=len(mol.GetAromaticAtoms()),
            # the next two elements are later used to compute the labels but will of course
            # not be used as features!
            mol_id=int(mol_id),
            atom_id=int(atom_idx),
            is_som=(atom_idx in soms),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=bond.GetBondTypeAsDouble(),
            bond_is_in_ring=bond.IsInRing(),
            bond_is_aromatic=bond.GetIsAromatic(),
            bond_is_conjugated=bond.GetIsConjugated(),
            bond_stereo=bond.GetStereo(),
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
        atomic_num = [int(current_node["atomic_num"])]
        degree = [int(current_node["degree"])]
        valence = [int(current_node["valence"])]
        formal_charge = [int(current_node["formal_charge"])]
        hybridization = [int(current_node["hybridization"])]
        num_hs = [int(current_node["num_hs"])]
        is_in_ring_3 = [int(current_node["is_in_ring_3"])]
        is_in_ring_4 = [int(current_node["is_in_ring_4"])]
        is_in_ring_5 = [int(current_node["is_in_ring_5"])]
        is_in_ring_6 = [int(current_node["is_in_ring_6"])]
        is_in_ring_7 = [int(current_node["is_in_ring_7"])]
        is_in_ring_8 = [int(current_node["is_in_ring_8"])]
        is_aromatic = [int(current_node["is_aromatic"])]
        vdw_radius = [float(current_node["vdw_radius"])]
        covalent_radius = [float(current_node["covalent_radius"])]
        num_h_acceptors = [float(current_node["num_h_acceptors"])]
        num_h_donors = [float(current_node["num_h_donors"])]
        num_rings = [float(current_node["num_rings"])]
        num_rotatable_bonds = [float(current_node["num_rotatable_bonds"])]
        molwt = [float(current_node["molwt"])]
        tpsa = [float(current_node["tpsa"])]
        logp = [float(current_node["logp"])]
        refractivity = [float(current_node["refractivity"])]
        labute_asa = [float(current_node["labute_asa"])]
        frac_rotatable_bonds = [float(current_node["frac_rotatable_bonds"])]
        num_O = [float(current_node["num_O"])]
        num_N = [float(current_node["num_N"])]
        num_S = [float(current_node["num_S"])]
        num_hal = [float(current_node["num_hal"])]
        num_sp3C = [float(current_node["num_sp3C"])]
        num_ar = [float(current_node["num_ar"])]

        features_vector = (
            atomic_num
            + degree
            + valence
            + formal_charge
            + hybridization
            + num_hs
            + is_in_ring_3
            + is_in_ring_4
            + is_in_ring_5
            + is_in_ring_6
            + is_in_ring_7
            + is_in_ring_8
            + is_aromatic
            + vdw_radius
            + covalent_radius
            + num_h_acceptors
            + num_h_donors
            + num_rings
            + num_rotatable_bonds
            + molwt
            + tpsa
            + logp
            + refractivity
            + labute_asa
            + frac_rotatable_bonds
            + num_O
            + num_N
            + num_S
            + num_hal
            + num_sp3C
            + num_ar
        )

        if i == 0:
            features = pd.DataFrame(
                {
                    "atomic_num": pd.Series(dtype="int"),
                    "degree": pd.Series(dtype="int"),
                    "valence": pd.Series(dtype="int"),
                    "formal_charge": pd.Series(dtype="int"),
                    "hybridization": pd.Series(dtype="int"),
                    "num_hs": pd.Series(dtype="int"),
                    "is_in_ring_3": pd.Series(dtype="int"),
                    "is_in_ring_4": pd.Series(dtype="int"),
                    "is_in_ring_5": pd.Series(dtype="int"),
                    "is_in_ring_6": pd.Series(dtype="int"),
                    "is_in_ring_7": pd.Series(dtype="int"),
                    "is_in_ring_8": pd.Series(dtype="int"),
                    "is_aromatic": pd.Series(dtype="int"),
                    "vdw_radius": pd.Series(dtype="float"),
                    "covalent_radius": pd.Series(dtype="float"),
                    "num_h_acceptors": pd.Series(dtype="float"),
                    "num_h_donors": pd.Series(dtype="float"),
                    "num_rings": pd.Series(dtype="float"),
                    "num_rotatable_bonds": pd.Series(dtype="float"),
                    "molwt": pd.Series(dtype="float"),
                    "tpsa": pd.Series(dtype="float"),
                    "logp": pd.Series(dtype="float"),
                    "refractivity": pd.Series(dtype="float"),
                    "labute_asa": pd.Series(dtype="float"),
                    "frac_rotatable_bonds": pd.Series(dtype="float"),
                    "num_O": pd.Series(dtype="float"),
                    "num_N": pd.Series(dtype="float"),
                    "num_S": pd.Series(dtype="float"),
                    "num_hal": pd.Series(dtype="float"),
                    "num_sp3C": pd.Series(dtype="float"),
                    "num_ar": pd.Series(dtype="float"),
                }
            )

        features.loc[len(features)] = features_vector

    features = features.astype(
        {
            "atomic_num": int,
            "degree": int,
            "valence": int,
            "formal_charge": int,
            "hybridization": int,
            "num_hs": int,
            "is_in_ring_3": int,
            "is_in_ring_4": int,
            "is_in_ring_5": int,
            "is_in_ring_6": int,
            "is_in_ring_7": int,
            "is_in_ring_8": int,
            "is_aromatic": int,
        }
    )

    categorical_columns_selector = make_column_selector(dtype_include=int)
    numerical_columns_selector = make_column_selector(dtype_include=float)

    numerical_columns = numerical_columns_selector(features)
    categorical_columns = categorical_columns_selector(features)

    categorical_preprocessor = OneHotEncoder(
        sparse=False, dtype=float, handle_unknown="infrequent_if_exist"
    )
    numerical_preprocessor = MinMaxScaler(feature_range=(0, 1))

    preprocessor = ColumnTransformer(
        [
            ("one-hot-encoder", categorical_preprocessor, categorical_columns),
            ("standard_scaler", numerical_preprocessor, numerical_columns),
        ]
    )

    features = preprocessor.fit_transform(features)

    return np.array(features)


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
    df["soms"] = df["soms"].map(ast.literal_eval)

    # Generate networkx graphs from mols and save them in a json file
    df["G"] = df.apply(lambda x: mol_to_nx(x.mol_id, x.ROMol, x.soms), axis=1)
    G = nx.disjoint_union_all(df["G"].to_list())
    with open(os.path.join(data_directory, "graph.json"), "w") as f:
        f.write(json.dumps(json_graph.node_link_data(G)))

    # Generate and save list of mol ids
    mol_ids = []
    for i in range(len(G.nodes)):
        mol_ids.append(G.nodes[i]["mol_id"])
    mol_ids = np.array(mol_ids)
    np.save(os.path.join(data_directory, "mol_ids.npy"), mol_ids)

    # Generate and save list of atom ids
    atom_ids = []
    for i in range(len(G.nodes)):
        atom_ids.append(G.nodes[i]["atom_id"])
    atom_ids = np.array(atom_ids)
    np.save(os.path.join(data_directory, "atom_ids.npy"), atom_ids)

    # Generate and save list of labels
    labels = []
    for i in range(len(G.nodes)):
        labels.append(int(G.nodes[i]["is_som"]))
    labels = np.array(labels)
    np.save(os.path.join(data_directory, "labels.npy"), labels)

    # Compute node features matrix and save it to node_features.npy
    node_features = compute_node_features_matrix(G)
    np.save(os.path.join(data_directory, "node_features.npy"), node_features)

    df = pd.DataFrame(node_features)
    corr_matrix = df.corr()
    plt.imshow(corr_matrix, cmap="binary")
    plt.savefig(os.path.join(data_directory, "correlation_matrix.png"))
