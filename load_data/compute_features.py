import torch
from tqdm import tqdm
import numpy as np
from rdkit import Chem

def binary_encoding(x, permitted_list):
    """
    Inputs an object x and a list of permitted values and returns a list of integers {0,1}
    encoding which of the permitted values the input object takes.

    Args:
        x (arbitrary): an object of arbitrary type
        permitted_list (list): a list of permitted values for x.

    Returns:
        x_encoded (list): a list of integers {0,1} encoding which of the permitted values the input object takes.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    x_encoded = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return x_encoded

def get_atom_features(atom):
    """
    Takes an RDKit atom object as input and returns a 1d-numpy array of atom features as output.
    Features:   atom type, degree, formal charge, hybridization, is in a ring, is aromatic, atomic mass, 
                Van der Waals radius, covalent radius, total number of Hs (explicit and implicit) on the atom.

    Args:
        atom (RDKit atom object): The atom for which to compute the features.

    Returns:
        atom_feature_vector (numpy.array): 1d NumPy array of length (number of featues).

    """

    permitted_list_of_atoms =  ['H', 'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si']
    
    atom_type = binary_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    formal_charge = binary_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridization_type = binary_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring = [int(atom.IsInRing())]
    is_aromatic = [int(atom.GetIsAromatic())]
    atomic_mass = [float(atom.GetMass())]
    vdw_radius = [float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))]
    covalent_radius = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())))]
    n_heavy_neighbors = binary_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    n_hydrogens = binary_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    atom_feature_vector = atom_type + formal_charge + hybridization_type + is_in_a_ring + is_aromatic + atomic_mass + vdw_radius + covalent_radius + n_heavy_neighbors + n_hydrogens

    return np.array(atom_feature_vector)

def features_from_mols(mols):
    """
    Takes a list of RDKit mol objects as input and returns an arry of shape (number of atoms, number of features)
    containing the features of all the atoms in the inputed list of mols.
    Args:
        mols (list of RDKit mol objects): the list of RDKit mol objects for which to predict the individual atom features.
    
    Returns:
        features (numpy.array): a 2d NumPy array of shape (number of atoms, number of features).
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
