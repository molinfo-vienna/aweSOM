import json
import logging
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from typing import List

__all__ = ["SOM"]


class SOM(InMemoryDataset):
    """Creates a PyTorch Geometric Dataset from preprocessed input data.
    The preprocessed data is generated from an input .sdf file via methods
    from the process_input_data.py file.
    Args:
        root (string): The directory where the dataset will be saved.
    """

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "graph.json",
            "node_features.npy",
            "mol_ids.npy",
            "atom_ids.npy",
            "labels.npy",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self) -> None:
        with open(self.root + "/graph.json", "r") as f:
            G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

        node_features = np.load(self.root + "/node_features.npy", allow_pickle=True)
        node_features = torch.from_numpy(node_features).to(torch.float)

        y = torch.from_numpy(np.load(self.root + "/labels.npy"))

        mol_ids = torch.from_numpy(np.load(self.root + "/mol_ids.npy")).to(torch.long)
        unique_mol_ids = torch.unique(mol_ids).tolist()

        atom_ids = torch.from_numpy(np.load(self.root + "/atom_ids.npy")).to(torch.long)

        data_list = []

        for mol_id in unique_mol_ids:
            try:
                mask = mol_ids == mol_id

                G_s = G.subgraph(np.flatnonzero(mask).tolist())

                edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
                edge_index_reset = edge_index - edge_index.min()

                for i, edge in enumerate(list(G_s.edges)):
                    if i == 0:
                        edge_attr = torch.empty(
                            (
                                len(G_s.edges),
                                len(
                                    G_s.get_edge_data(edge[0], edge[1])["bond_features"]
                                ),
                            )
                        )
                    edge_attr[i] = torch.tensor(
                        G_s.get_edge_data(edge[0], edge[1])["bond_features"]
                    )

                # num_subsamplings = 30  # this is a hyperparameter
                # sampling_mask = torch.empty((len(y[mask]), num_subsamplings))

                # neg = (y[mask] == False).nonzero(as_tuple=True)[0]
                # num_negs = min(3, len(neg))  # this is a hyperparameter

                # for i in range(num_subsamplings):
                #     sub_neg = neg[torch.randperm(len(neg))[:num_negs]]
                #     sub = y[mask]
                #     for index in sub_neg:
                #         sub[index] = 1
                #     sampling_mask[:, i] = sub

                data = Data(
                    x=node_features[mask],
                    edge_index=edge_index_reset,
                    edge_attr=edge_attr,
                    y=y[mask],
                    # sampling_mask=sampling_mask,
                    mol_id=torch.full((len(y[mask]),), mol_id),
                    atom_id=atom_ids[mask],
                )

                data_list.append(data)

            except Exception as e:
                logging.warning(
                    f"An error occurred on molecule with mol_id {mol_id}. Exception: {e}"
                )
                continue

        torch.save(self.collate(data_list), self.processed_paths[0])
