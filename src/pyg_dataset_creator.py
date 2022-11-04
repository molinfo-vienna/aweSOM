import json
import logging
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

class SOM(InMemoryDataset):
    """This dataset is created from a preprocessed list of small molecules
    obtained from the MetaQSAR data set, which is a manually compiled resource
    of published measured data on xenobiotic metabolism, including expert-curated
    SoMs and reaction annotations for discovery compounds and drugs.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['node_features.npy', 'mol_ids.npy', 'graph.json', 'labels.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        with open('data/graph.json', 'r') as f:
            G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

        node_features = np.load('data/node_features.npy', allow_pickle=True)
        node_features = torch.from_numpy(node_features).to(torch.float)

        y = np.load('data/labels.npy')
        y = torch.from_numpy(y)

        mol_ids = torch.from_numpy(np.load('data/mol_ids.npy')).to(torch.long)
        unique_mol_ids = torch.unique(mol_ids).tolist()

        data_list = []

        for mol_id in (unique_mol_ids):
            try:
                mask = mol_ids == mol_id  # mask is a tensor of trues and falses showing where mol_ids is equal to mol_id

                G_s = G.subgraph(np.flatnonzero(mask).tolist())  # select a subgraph G_s from G corresponding to the atoms from the mol with the current mol_id

                edge_index = torch.tensor(list(G_s.edges)).t().contiguous()  # gets a tensor containing the edge indices from the OutEdgeView representation
                edge_index_reset = edge_index - edge_index.min()  # resets the edges labeling within a molecular graph (every edge_index tensor starts with node 0)

                edge_attr = torch.empty(len(G_s.edges), 5)
                for i, edge in enumerate(list(G_s.edges)):
                    edge_attr[i, 0] = G_s.get_edge_data(edge[0], edge[1])['bond_type']
                    edge_attr[i, 1] = G_s.get_edge_data(edge[0], edge[1])['bond_is_in_ring']
                    edge_attr[i, 2] = G_s.get_edge_data(edge[0], edge[1])['bond_is_aromatic']
                    edge_attr[i, 3] = G_s.get_edge_data(edge[0], edge[1])['bond_is_conjugated']
                    edge_attr[i, 4] = G_s.get_edge_data(edge[0], edge[1])['bond_stereo']

                num_subsamplings = 50  # this is a hyperparameter
                sampling_mask = torch.empty((len(y[mask]), num_subsamplings))

                neg = (y[mask]==False).nonzero(as_tuple=True)[0]
                num_negs = min(3, len(neg))  # this is a hyperparameter

                for i in range(num_subsamplings):
                    sub_neg = neg[torch.randperm(len(neg))[:num_negs]]
                    sub = y[mask]
                    for index in sub_neg: sub[index] = 1
                    sampling_mask[:,i] = sub

                data = Data(x=node_features[mask], edge_index=edge_index_reset, edge_attr=edge_attr, y=(y[mask]), sampling_mask=sampling_mask)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                   data = self.pre_transform(data)

                data_list.append(data)
            except:
                logging.warning(f"An error occurred on molecule with mol_id {mol_id}.")

        torch.save(self.collate(data_list), self.processed_paths[0])
