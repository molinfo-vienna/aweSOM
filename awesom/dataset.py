import numpy as np
import os
import pandas as pd
import torch

from ast import literal_eval
from multiprocessing import cpu_count
from rdkit.Chem import PandasTools
from torch_geometric.data import InMemoryDataset, Data
from typing import List

from awesom.dataset_utils import (
    _get_file_length,
    generate_preprocessed_data_RDKit,
    generate_preprocessed_data_CDPKit,
)


KIT = "RDKIT"


class SOM(InMemoryDataset):
    """Creates a PyTorch Geometric Dataset from data.sdf.
    The files must contain a <soms> attribute that is a list of the atom indices (start 0)
    that are Sites of Metabolism.
    Args:
        root (string): The directory where the input data is stored (root/raw) and to which
            the preprocessed dataset will be saved (root/preprocessed).
    """

    def __init__(
        self, root: str, transform=None, pre_transform=None, pre_filter=None
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> None:
        pass

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        input_path = os.path.join(
            os.path.join(self.root, "raw"),
            os.listdir(os.path.join(self.root, "raw"))[0],
        )
        data_list = self.data_processing(input_path=input_path, labels=True)
        torch.save((self.collate(data_list)), self.processed_paths[0])

    def data_processing(self, input_path, labels):
        _, file_extension = os.path.splitext(input_path)

        if KIT == "RDKIT":
            if file_extension == ".sdf":
                df = PandasTools.LoadSDF(input_path, removeHs=False)
            elif file_extension == ".smi":
                df = pd.read_csv(input_path, names=["smiles"])
                PandasTools.AddMoleculeColumnToFrame(df, "smiles")
            else:
                raise NotImplementedError(f"Invalid file extension: {file_extension}")

            if "ID" not in df:
                df["ID"] = df.index
            if labels:
                # If there is SoM information, check if every entry has a non-empty list of SoMs,
                # and if not remove that entry from the dataframe and print a warning
                df["soms"] = df["soms"].map(literal_eval)
                df_out = df[df["soms"].map(len) == 0]
                if len(df_out) > 0:
                    print(
                        f"Warning: {len(df_out)} entries have no SoMs and will be removed from the dataset."
                    )
                    for i in df_out["ID"]:
                        print(f"Entry {i} has no SoMs.")
                df = df[df["soms"].map(len) > 0]
            else:
                # If there is no SoM information, create an empty list for each entry
                df["soms"] = "[]"
                df["soms"] = df["soms"].map(literal_eval)

            G = generate_preprocessed_data_RDKit(df, min(len(df), cpu_count()))

        elif KIT == "CDPKIT":
            file_length = _get_file_length(input_path)
            G = generate_preprocessed_data_CDPKit(
                input_path, file_length, min(file_length, cpu_count()), labels
            )

        else:
            raise IOError("Error: invalid chemistry toolkit name")

        return self.create_data_list(G)

    def create_data_list(self, G):
        # Compute list of mol ids
        mol_ids = torch.from_numpy(
            np.array([G.nodes[i]["mol_id"] for i in range(len(G.nodes))])
        ).to(torch.long)

        # Compute list of atom ids
        atom_ids = torch.from_numpy(
            np.array([G.nodes[i]["atom_id"] for i in range(len(G.nodes))])
        ).to(torch.long)

        # Compute list of labels
        labels = torch.from_numpy(
            np.array([int(G.nodes[i]["is_som"]) for i in range(len(G.nodes))])
        )

        # Compute node features matrix
        num_nodes = len(G.nodes)
        node_features = np.empty((num_nodes, len(G.nodes()[0]["node_features"])))
        for i in range(num_nodes):
            current_node = G.nodes[i]
            node_features[i, :] = current_node["node_features"]
        node_features = torch.from_numpy(node_features).to(torch.float)

        # Compute mol features matrix
        # mol_features = np.empty((num_nodes, len(G.nodes()[0]["mol_features"])))
        # for i in range(num_nodes):
        #     current_node = G.nodes[i]
        #     mol_features[i, :] = current_node["mol_features"]
        # mol_features = torch.from_numpy(mol_features).to(torch.float)

        data_list = []

        for mol_id in torch.unique(mol_ids).tolist():
            try:
                mask = mol_ids == mol_id

                subG = G.subgraph(np.flatnonzero(mask).tolist())

                edge_index = torch.tensor(list(subG.edges)).t().contiguous()
                edge_index_reset = edge_index - edge_index.min()

                for i, edge in enumerate(list(subG.edges)):
                    if i == 0:
                        edge_attr = torch.empty(
                            (
                                len(subG.edges),
                                len(
                                    subG.get_edge_data(edge[0], edge[1])[
                                        "bond_features"
                                    ]
                                ),
                            )
                        )
                    edge_attr[i] = torch.tensor(
                        subG.get_edge_data(edge[0], edge[1])["bond_features"]
                    )

                data = Data(
                    x=node_features[mask],
                    edge_index=edge_index_reset,
                    edge_attr=edge_attr,
                    # mol_x=mol_features[mask],
                    y=labels[mask],
                    mol_id=torch.full((labels[mask].shape[0],), mol_id),
                    atom_id=atom_ids[mask],
                )

                data_list.append(data)

            except Exception as e:
                print(f"An error occurred on molecule with mol_id {mol_id}:", e)
                continue

        return data_list

    def get_pos_weight(self):
        total_num_atoms = 0
        num_soms = 0
        for data in self:
            for label in data.y:
                total_num_atoms += 1
                num_soms += int(label)
        num_nosom = total_num_atoms - num_soms
        return num_nosom / num_soms


class LabeledData(SOM):
    """Creates a PyTorch Geometric Dataset from a data.sdf input data file.
    Args:
        root (string): The directory where the input data is stored (root/raw) and to which
            the preprocessed dataset will be saved (root/preprocessed).
    """

    def __init__(
        self, root: str, transform=None, pre_transform=None, pre_filter=None
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> None:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self):
        file_name = os.listdir(os.path.join(self.root, "raw"))[0]
        if file_name.endswith(".smi") or file_name.endswith(".sdf"):
            input_path = os.path.join(
                os.path.join(self.root, "raw"),
                os.listdir(os.path.join(self.root, "raw"))[0],
            )
        else:
            raise NotImplementedError(
                'Data file must be either "data.smi" or "data.sdf".'
            )
        data_list = self.data_processing(input_path=input_path, labels=True)
        torch.save((self.collate(data_list)), self.processed_paths[0])


class UnlabeledData(SOM):
    """Creates a PyTorch Geometric Dataset from a data.sdf input data file.
    Args:
        root (string): The directory where the input data is stored (root/raw) and to which
            the preprocessed dataset will be saved (root/preprocessed).
    """

    def __init__(
        self, root: str, transform=None, pre_transform=None, pre_filter=None
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> None:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self):
        file_name = os.listdir(os.path.join(self.root, "raw"))[0]
        if file_name.endswith(".smi") or file_name.endswith(".sdf"):
            input_path = os.path.join(
                os.path.join(self.root, "raw"),
                os.listdir(os.path.join(self.root, "raw"))[0],
            )
        else:
            raise NotImplementedError(
                'Data file must be either "data.smi" or "data.sdf".'
            )
        data_list = self.data_processing(input_path=input_path, labels=False)
        torch.save((self.collate(data_list)), self.processed_paths[0])
