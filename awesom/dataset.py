import ast
import networkx as nx
import numpy as np
import os
import pandas as pd
import torch

from multiprocessing import cpu_count
from rdkit.Chem import PandasTools
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import add_remaining_self_loops
from typing import List

from awesom.dataset_utils import (
    _get_file_length,
    generate_preprocessed_data_RDKit,
    generate_preprocessed_data_CDPKit,
)


__all__ = ["SOM", "LabeledData", "UnlabeledData"]


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
    def raw_file_names(self) -> List[str]:
        return ["data.sdf"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        data_list = self.data_processing(path=self.raw_paths[0], labels=True)
        torch.save((self.collate(data_list)), self.processed_paths[0])

    def data_processing(self, path, labels):
        _, file_extension = os.path.splitext(path)

        if KIT == "RDKIT":
            if file_extension == ".sdf":
                df = PandasTools.LoadSDF(path, removeHs=True)
            elif file_extension == ".smi":
                df = pd.read_csv(path, names=["smiles"])
                PandasTools.AddMoleculeColumnToFrame(df, "smiles")
            else:
                raise NotImplementedError(f"Invalid file extension: {file_extension}")
            
            if 'ID' not in df:
                df["ID"] = df.index
            if not labels:
                df["soms"] = "[]"
            df["soms"] = df["soms"].map(ast.literal_eval)
            #################################################################
            df["reamainclasses"] = df["reamainclasses"].map(ast.literal_eval)
            df["reaclasses"] = df["reaclasses"].map(ast.literal_eval)
            df["reasubclasses"] = df["reasubclasses"].map(ast.literal_eval)
            #################################################################

            G = generate_preprocessed_data_RDKit(df, min(len(df), cpu_count()))

        elif KIT == "CDPKIT":
            file_length = _get_file_length(path)
            G = generate_preprocessed_data_CDPKit(
                path, file_length, min(file_length, cpu_count()), labels
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
        # labels = torch.transpose(torch.from_numpy(
        #     np.array(
        #         [
        #             [int(not G.nodes[i]["is_som"]) for i in range(len(G.nodes))],
        #             [int(G.nodes[i]["is_som"]) for i in range(len(G.nodes))],
        #         ]
        #     )
        # ), 0, 1)

        # Compute node features matrix
        num_nodes = len(G.nodes)
        node_features = np.empty((num_nodes, len(G.nodes()[0]["node_features"])))
        for i in range(num_nodes):
            current_node = G.nodes[i]
            node_features[i, :] = current_node["node_features"]
        node_features = torch.from_numpy(node_features).to(torch.float)

        # # Compute coordinates matrix
        # num_nodes = len(G.nodes)
        # coordinates = torch.zeros((num_nodes, 3))
        # for i in range(num_nodes):
        #     current_node = G.nodes[i]
        #     coordinates[i, :] = current_node["coordinates"]

        data_list = []

        for mol_id in torch.unique(mol_ids).tolist():

            try:
                mask = mol_ids == mol_id

                subG = G.subgraph(np.flatnonzero(mask).tolist())

                edge_index = torch.tensor(list(subG.edges)).t().contiguous()
                edge_index_reset = edge_index - edge_index.min()
                # edge_index_reset = add_remaining_self_loops(edge_index_reset)[0]

                # complete_graph = nx.complete_graph(len(subG.nodes))
                # fully_connected_edges = torch.tensor(list(complete_graph.edges - subG.edges)).t().contiguous()

                # edge_index_reset = torch.cat((edge_index_reset, fully_connected_edges), dim=1)

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

                # edge_attr = torch.cat((edge_attr, torch.zeros((fully_connected_edges.shape[1], edge_attr.shape[1]))))

                data = Data(
                    x=node_features[mask],
                    edge_index=edge_index_reset,
                    edge_attr=edge_attr,
                    y=labels[mask],
                    mol_id=torch.full((labels[mask].shape[0],), mol_id),
                    atom_id=atom_ids[mask],
                    # pos=coordinates[mask]
                )

                data_list.append(data)

            except Exception as e:
                print(f"An error occurred on molecule with mol_id {mol_id}:", e)
                continue

        return data_list

    def get_class_distribution(self):
        size = 0
        som = 0
        for data in self:
            for label in data.y:
                size += 1
                som += int(label)
        nosom = size - som
        return size, nosom, som

    def get_class_weights(self):
        size, nosom, som = self.get_class_distribution()
        return torch.tensor(
            [size / (2 * nosom), (size / (2 * som))], dtype=torch.float
        ).cuda()


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
    def raw_file_names(self) -> List[str]:
        for file in os.listdir(os.path.join(self.root, "raw")):
            if file.endswith(".smi") or file.endswith(".sdf"):
                return [str(file)]
            else:
                raise NotImplementedError(
                    'Data file must be either "data.smi" or "data.sdf".'
                )

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self):
        data_list = self.data_processing(path=self.raw_paths[0], labels=True)
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
    def raw_file_names(self) -> List[str]:
        for file in os.listdir(os.path.join(self.root, "raw")):
            if file.endswith(".smi") or file.endswith(".sdf"):
                return [str(file)]
            else:
                raise NotImplementedError(
                    'Data file must be either "data.smi" or "data.sdf".'
                )

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self):
        data_list = self.data_processing(path=self.raw_paths[0], labels=False)
        torch.save((self.collate(data_list)), self.processed_paths[0])
