import json
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
        pre_trasnform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        - #graphs: 2216
        - #node_features: 48
        - #classes: 2
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['features.npy', 'mol_ids.npy', 'graph.json', 'labels.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        with open('load_data/data/graph.json', 'r') as f:
            G = nx.Graph(json_graph.node_link_graph(json.load(f)))

        x = np.load('load_data/data/features.npy')
        x = torch.from_numpy(x).to(torch.float)

        y = np.load('load_data/data/labels.npy')
        y = torch.from_numpy(y).to(torch.long)

        mol_ids = torch.from_numpy(np.load('load_data/data/mol_ids.npy')).to(torch.long)
        unique_mols_ids = torch.unique(mol_ids).tolist()

        data_list = []

        for mol_id in (unique_mols_ids):
            try:
                mask = mol_ids == mol_id  # mask is an array of trues and falses showing where mol_ids is equal to mol_id
                G_s = G.subgraph(np.flatnonzero(mask).tolist())  # select a subgraph G_s from G corresponding to the atoms from the mol with the current mol_id
                edge_index = torch.tensor(list(G_s.edges)).t().contiguous()  # gets a tensor containing the edge indices from the OutEdgeView representation
                edge_index = edge_index - edge_index.min()  # resets the edges labeling within a molecular graph (every edge_index tensor starts with node 0)
                #edge_index, _ = remove_self_loops(edge_index)

                data = Data(edge_index=edge_index, x=x[mask], y=(y[mask]).to(torch.long))

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                   data = self.pre_transform(data)

                data_list.append(data)
            except:
                print("An error occurred on molecule ", mol_id)

        torch.save(self.collate(data_list), self.processed_paths[0])
