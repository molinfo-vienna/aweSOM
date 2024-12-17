import os
import shutil
from ast import literal_eval
from multiprocessing import cpu_count
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import PandasTools
from torch_geometric.data import Data, InMemoryDataset

from awesom.dataset_utils import(
    generate_preprocessed_data, 
    remove_implicit_Hs,
)


class SOM(InMemoryDataset):
    """Base class to create a PyTorch Geometric Dataset from and SD-File or an smiles file."""

    def __init__(
        self, root: str, transform=None, pre_transform=None, pre_filter=None
    ) -> None:
        # # Delete the processed folder if it exists
        # processed_folder = os.path.join(root, "processed")
        # if os.path.exists(processed_folder):
        #     shutil.rmtree(processed_folder)
        #     print(f"Deleted existing processed folder at: {processed_folder}")

        # Call the superclass constructor
        super().__init__(root, transform, pre_transform, pre_filter)

        self.root = root
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def find_input_file(
        self, extensions: List[str] = [".sdf", ".smi", ".smiles"]
    ) -> Optional[str]:
        """Finds the first file in the root directory with one of the allowed extensions."""
        for file_name in os.listdir(self.root):
            if any(file_name.endswith(ext) for ext in extensions):
                return os.path.join(self.root, file_name)
        return None

    def process(self, labels: bool = True) -> None:
        input_file = self.find_input_file()
        if input_file is None:
            raise NotImplementedError(
                "Data file must be either .sdf, .smi, or .smiles."
            )

        data_list = self.data_processing(input_file=input_file, labels=labels)
        torch.save((self.collate(data_list)), self.processed_paths[0])

    def data_processing(self, input_file: str, labels: bool) -> List[Data]:
        _, file_extension = os.path.splitext(input_file)

        # Load data from the file
        if file_extension == ".sdf":
            # Load the SD-File without removing the hydrogen atoms.
            # Hydrogens are removed later in the process,
            # by taking care of re-assigning the correct SOM-atom indices.
            df = PandasTools.LoadSDF(input_file, removeHs=False)
        elif file_extension == ".smi" or file_extension == ".smiles":
            df = pd.read_csv(input_file, names=["smiles"])
            PandasTools.AddMoleculeColumnToFrame(df, "smiles")
        else:
            raise NotImplementedError(f"Invalid file extension: {file_extension}")

        # Set an ID column if not already present
        df["ID"] = df.get("ID", df.index).astype(str)

        # Process SoM information based on the presence of labels
        if labels:
            # Ensure the "soms" column is parsed as lists
            df["soms"] = df["soms"].map(literal_eval)

            # Identify and warn about entries without SoMs
            no_som_entries = df[df["soms"].map(len) == 0]
            if not no_som_entries.empty:
                print(f"Warning: {len(no_som_entries)} entries have no SoMs.")
                for mol_id in no_som_entries["ID"]:
                    print(f"Entry with ID {mol_id} has no SoMs.")

            # Filter out entries without SoMs
            df = df[df["soms"].map(len) > 0].reset_index(drop=True)
        else:
            # If no SoM info is available, initialize empty lists for each molecule
            df["soms"] = [[] for _ in range(len(df))]

        # Remove implicit hydrogens
        df[['ROMol', 'soms']] = df.apply(remove_implicit_Hs, axis=1, result_type='expand')

        # Set a numerical (integer) mol_id for each molecule
        df["mol_id"] = df.index

        # Generate preprocessed data
        G = generate_preprocessed_data(df, min(len(df), cpu_count()))
        return self.create_data_list(G)

    def create_data_list(self, G) -> List[Data]:
        """Creates a list of Data objects from a graph object G."""
        mol_ids = torch.tensor(
            [G.nodes[i]["mol_id"] for i in range(len(G.nodes))], dtype=torch.int32
        )
        atom_ids = torch.tensor(
            [G.nodes[i]["atom_id"] for i in range(len(G.nodes))], dtype=torch.int32
        )
        labels = torch.tensor(
            [int(G.nodes[i]["is_som"]) for i in range(len(G.nodes))], dtype=torch.int32
        )
        ids = [G.nodes[i]["id"] for i in range(len(G.nodes))]
        node_features = torch.tensor(
            [G.nodes[i]["node_features"] for i in range(len(G.nodes))],
            dtype=torch.float32,
        )

        # # Compute mol features matrix
        # mol_features = torch.tensor(
        #             [G.nodes[i]["mol_features"] for i in range(len(G.nodes))], dtype=torch.float
        #         )

        data_list = []

        for mol_id in mol_ids.unique():
            try:
                mask = mol_ids == mol_id.item()
                subG = G.subgraph(np.flatnonzero(mask.numpy()).tolist())

                edge_index = torch.tensor(list(subG.edges)).t().contiguous()
                edge_index_reset = edge_index - edge_index.min()
                edge_attr = torch.tensor(
                    [subG.get_edge_data(*edge)["bond_features"] for edge in subG.edges],
                    dtype=torch.float,
                )

                data = Data(
                    x=node_features[mask],
                    edge_index=edge_index_reset,
                    edge_attr=edge_attr,
                    # mol_x=mol_features[mask],
                    y=labels[mask],
                    mol_id=mol_id,
                    atom_id=atom_ids[mask],
                )

                data.description = [d for d, m in zip(ids, mask) if m][0]

                data_list.append(data)

            except Exception as e:
                print(f"An error occurred on molecule with mol_id {mol_id.item()}:", e)
                continue

        return data_list


class LabeledData(SOM):
    """Class to create a PyTorch Geometric Dataset from and SD-File or an smiles file with SOM-labels."""

    def process(self):
        super().process(labels=True)


class UnlabeledData(SOM):
    """Class to create a PyTorch Geometric Dataset from and SD-File or an smiles file without SOM-labels."""

    def process(self):
        super().process(labels=False)
