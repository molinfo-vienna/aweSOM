from ast import literal_eval
from os import PathLike
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset


class SOMDataset(Dataset):
    """PyTorch Geometric Dataset for site-of-metabolism prediction from SD-Files or SMILES files."""

    def __init__(
        self,
        input_path: PathLike[str] | str,
        error: Literal["raise", "drop"] = "drop",
    ) -> None:
        super().__init__(str(input_path))
        self.data = [
            mol_to_data(mol, soms, i, description)
            for i, (mol, soms, description) in enumerate(
                self.load_molecules(Path(input_path), error)
            )
        ]

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int) -> Data:
        return self.data[idx]

    def load_molecules(
        self,
        input_file: Path,
        error: Literal["raise", "drop"],
    ) -> list[tuple[Chem.Mol, list[int], str]]:
        """Load molecules from file based on extension."""
        results = []

        if input_file.suffix == ".sdf":
            suppl = Chem.SDMolSupplier(str(input_file), removeHs=False)
            for mol_num, mol in enumerate(suppl):
                if mol is None:
                    if error == "raise":
                        raise RuntimeError(
                            f"Could not parse molecule {mol_num} in input"
                        )
                    elif error == "drop":
                        continue

                props = {
                    key.lower(): value for key, value in mol.GetPropsAsDict().items()
                }

                soms = literal_eval(props.get("soms", "[]"))
                desc = props.get("_name", f"{mol_num}")
                results.append((mol, soms, desc))

        elif input_file.suffix in [".smi", ".smiles"]:
            with input_file.open("r") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Parse SMILES (assuming format: SMILES\tID\tSoMs)
                    parts = line.split("\t")
                    smiles = parts[0]

                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        if error == "raise":
                            raise RuntimeError(
                                f"Could not parse molecule {line_num} in input"
                            )
                        elif error == "drop":
                            continue

                    if len(parts) > 2:
                        soms_str = parts[2]
                        soms = literal_eval(soms_str)
                    else:
                        soms = []

                    desc = parts[1] if len(parts) > 1 else f"{line_num}"

                    results.append((mol, soms, desc))
        else:
            raise NotImplementedError(f"Invalid file extension: {input_file.suffix}")

        return results


_to_undirected = T.ToUndirected()


def mol_to_data(mol: Chem.Mol, soms: list[int], mol_id: int, description: str) -> Data:
    """Convert a molecule to a PyTorch Geometric Data object."""
    # Generate atom features
    mol, soms = _remove_hydrogens_and_update_soms(mol, soms)

    atom_features = []
    atom_ids = []
    som_labels = []

    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        features = _get_atom_features(atom)
        atom_features.append(features)
        atom_ids.append(atom_id)
        som_labels.append(1 if atom_id in soms else 0)

    # Generate bond features and edge indices
    edge_index_list = []
    edge_attr_list = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edge_index_list.append([begin_idx, end_idx])
        bond_features = _get_bond_features(bond)
        edge_attr_list.extend([bond_features])

    # Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    y = torch.tensor(som_labels, dtype=torch.long)
    mol_ids = torch.full((len(atom_ids),), mol_id, dtype=torch.long)
    atom_ids_tensor = torch.tensor(atom_ids, dtype=torch.long)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        mol_id=mol_ids,
        atom_id=atom_ids_tensor,
        description=description,
        smiles=Chem.MolToSmiles(mol),
    )
    data.description = description

    data = _to_undirected.forward(data)
    assert isinstance(data, Data)

    return data


def _remove_hydrogens_and_update_soms(
    mol: Chem.Mol, soms: list[int]
) -> tuple[Chem.Mol, list[int]]:
    """Remove hydrogens and update SoM indices."""
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        if atom_id in soms:
            atom.SetIntProp("label", 1)
        else:
            atom.SetIntProp("label", 0)

    mol_no_h = Chem.RemoveHs(mol)

    new_soms = []
    for atom in mol_no_h.GetAtoms():
        if atom.GetIntProp("label") == 1:
            new_soms.append(atom.GetIdx())

    return mol_no_h, new_soms


def _get_atom_features(atom: Chem.Atom) -> list[float]:
    """Generate atom features."""
    atomic_num = atom.GetAtomicNum()
    element_list = [
        5,  # B
        6,  # C
        7,  # N
        8,  # O
        9,  # F
        14,  # Si
        15,  # P
        16,  # S
        17,  # Cl
        35,  # Br
        53,  # I
    ]

    features = []
    for element in element_list:
        features.append(1.0 if atomic_num == element else 0.0)
    features.append(1.0 if atomic_num not in element_list else 0.0)

    return features


def _get_bond_features(bond: Chem.Bond) -> list[float]:
    """Generate bond features."""
    bond_types = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    bond_type_str = str(bond.GetBondType())

    features = []
    for bond_type in bond_types:
        features.append(1.0 if bond_type_str == bond_type else 0.0)
    features.append(1.0 if bond_type_str not in bond_types else 0.0)

    features.append(1.0 if bond.IsInRing() else 0.0)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)

    return features
