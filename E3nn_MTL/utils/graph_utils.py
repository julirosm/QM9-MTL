import numpy as np
import pandas as pd
import os
from itertools import chain

import torch

from mendeleev import element

from torch_geometric.data import InMemoryDataset, Data
import torch.nn.functional as F
import torch_scatter

def create_node_features(elements, properties):
    """
    Creates a dictionary which asigns to each element its node features

    Parameters
    ----------
    elements (list): List containing the elements(str) which will be on
        on the list.
    properties (list): List containing physical features of nodes. Features must correspond to valid
          mendeleev.Element keys (see `mendeleev <https://mendeleev.readthedocs.io/en/stable/>`_) 


    Returns
    -------
    node_dict (pandas.DataFrame): Dataframe whose columns are the elements and rows the physical
    properties. In practice, it works as a vectorized dictionary.

    """
    n_elements = len(elements)
    n_properties = len(properties)
    properties_array = np.zeros((n_elements, n_properties))
    
    for i, e in enumerate(elements):
        elem = element(e)
        for j, prop in enumerate(properties):
            p = getattr(elem, prop)
            if callable(p):
                p = p()
            elif prop=="group":
                p = p.group_id
            properties_array[i,j] = p
    
    properties_df = pd.DataFrame(data = properties_array, index = elements, columns=properties)
        
    return properties_df

def energy_conversion(currents_units, desired_units):
    if currents_units == "Ha" and desired_units == "eV":
        return 27.211386245988
    elif currents_units == "eV" and desired_units == "Ha":
        return 1 / 27.211386245988
    elif currents_units == "Ha" and desired_units == "Ha":
        return 1
    elif currents_units == "eV" and desired_units == "eVs":
        return 1
    else:
        raise ValueError(f"Unsupported conversion: {currents_units} -> {desired_units}")




def graph_maker(file, properties_df, r_max, units="eV"):
    """
 Constructs a molecular graph from a QM9 .xyz file. Node features are introduced as input. Edge features are
 a gaussian histogram exp(-(dij-mu_t)^2/sigma^2) where mu_t = mu*t and t runs from 0 to n_bins

 Args:
 - file (str): Path to the .xyz file.
 - properties_df(pandas.DataFrame): Pandas dataframe (this works as a vectorized dictionary) which assigns to 
 each atom type the desired proporty. This can easily be created with the create_node_features
 function.
 - bonds_df(pandas.DataFrame): Pandas dataframe which assigns to each atom its covalent radius (in pm).
 - histogram(list): Three-dimensional list which contains mean, std, and number of bins of the
 gaussian histogram. 

 Returns:
 - graph (torch_geometric.data.Data): The PyG graph object.
 """       
    
    ### 1.Open file 
    with open(file, "r") as molec_file:
        lines=molec_file .readlines()
        
    # First row is the number of atoms in the molecule
    n_atoms = int(lines[0])
     
    # Dataframe with positions and element of each molecule
    atoms = pd.read_table(file, skiprows=list(chain(range(0, 2), range(n_atoms + 2, n_atoms + 5))) , sep='\s+', names = ['atom', 'x', 'y', 'z', 'e'])
     
    # Dictionary, Element:Atomic number (Z)
    atom_df = pd.DataFrame(data = np.array([1,6,7,8,9]).reshape(1,-1),columns=["H", "C", "N", "O", "F"])
     
    ### 2. Create variables for graph: Node features, Edge Index, ...
    ## Node features:
    x = torch.tensor(properties_df.loc[atoms["atom"]].values, dtype=torch.float)
    pos = torch.tensor(atoms[["x","y","z"]].values, dtype=torch.float)
    Z = torch.tensor(atom_df[atoms["atom"]].T.values)
    SMILES = lines[n_atoms+3].rsplit('\t')[0]
        
    ## Edges
    A = torch.zeros([n_atoms, n_atoms], dtype=torch.float)              # Adjacency matrix
     
    # Distance calculation
    diff = pos.unsqueeze(1) - pos.unsqueeze(0) 
    distances = torch.sqrt(torch.sum(diff**2,dim=-1))

    # Adjacency matrix
    A = (distances <= r_max)
    edge_index = torch.vstack([torch.nonzero(A)[:,0].view(1,-1),torch.nonzero(A)[:,1].view(1,-1)])
     
    ## Features
    y = torch.tensor(np.fromstring(lines[1].partition("\t")[2], sep="\t"), dtype=torch.float)
     
    ### 3.Creating PyG graph
    graph = Data(x=x.float(), pos=pos.float(), edge_index=edge_index.to(torch.int64), y=y.float(), z=Z.int(), Z=Z.int(), n_atoms = n_atoms, SMILES = SMILES)

    return graph


class QM9Dataset(InMemoryDataset):
    """
    QM9 dataset class.

    Given a folder containing `.xyz` files for molecules, this class creates a dataset
    of PyTorch Geometric (PyG) graphs saved as `QM9.pt`. The `.xyz` files should be located
    in a subfolder named `/raw` within the provided directory.

    The dataset includes the following molecular properties:

    | Index | Property               | Unit        | Description                                     |
    |-------|------------------------|-------------|-------------------------------------------------|
    | 0     | \( A \)                | GHz         | Rotational constant                             |
    | 1     | \( B \)                | GHz         | Rotational constant                             |
    | 2     | \( C \)                | GHz         | Rotational constant                             |
    | 3     | \( \mu \)              | D           | Dipole moment                                   |
    | 4     | \( \alpha \)           | \( a_0^3 \) | Isotropic polarizability                        |
    | 5     | \( \epsilon_{HOMO} \)  | eV          | Energy of HOMO                                  |
    | 6     | \( \epsilon_{LUMO} \)  | eV          | Energy of LUMO                                  |
    | 7     | \( \epsilon_{gap} \)   | eV          | Gap (\( \epsilon_{LUMO} - \epsilon_{HOMO} \))   |
    | 8     | \( \langle R^2 \rangle \) | \( a_0^2 \) | Electronic spatial extent                     |
    | 9     | \( zpve \)             | eV          | Zero point vibrational energy                   |
    | 10    | \( U_o \)              | eV          | Internal energy at 0 K                          |
    | 11    | \( U \)                | eV          | Internal energy at 298.15 K                     |
    | 12    | \( H \)                | eV          | Enthalpy at 298.15 K                            |
    | 13    | \( G \)                | eV          | Free energy at 298.15 K                         |
    | 14    | \( C_v \)              | cal/mol·K   | Heat capacity at 298.15 K                       |
    """

    def __init__(self, root, properties_df, r_max, transform=None, pre_transform=None):
        self.properties_df = properties_df
        self.bonds_df = create_node_features(["H", "C", "N", "O", "F"], ["covalent_radius_cordero"])
        self.r_max = r_max
        super().__init__(root, transform, pre_transform)
        
        # Load the processed data
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith(".xyz")]

    @property
    def processed_file_names(self):
        return ['QM9.pt']

    def download(self):
        pass

    def process(self):  
        data_list = []
        for raw_path in self.raw_file_names:
            file_path = os.path.join(self.raw_dir, raw_path)
            graph = graph_maker(file=file_path,
                                properties_df=self.properties_df,
                                r_max=self.r_max)

            data_list.append(graph)

        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data

    
class SetTarget(object):
    def __init__(self, target=0):
        self.target = target

    def __call__(self, data):
        data.y = data.y[self.target]
        return data

def create_properties_dict():
    return {
        0: {'Property': 'A', 'Unit': 'GHz', 'Description': 'Rotational constant'},
        1: {'Property': 'B', 'Unit': 'GHz', 'Description': 'Rotational constant'},
        2: {'Property': 'C', 'Unit': 'GHz', 'Description': 'Rotational constant'},
        3: {'Property': 'μ', 'Unit': 'D', 'Description': 'Dipole moment'},
        4: {'Property': 'α', 'Unit': 'a₀³', 'Description': 'Isotropic polarizability'},
        5: {'Property': 'ε_HOMO', 'Unit': 'eV', 'Description': 'Energy of HOMO'},
        6: {'Property': 'ε_LUMO', 'Unit': 'eV', 'Description': 'Energy of LUMO'},
        7: {'Property': 'ε_gap', 'Unit': 'eV', 'Description': 'Gap (ε_LUMO - ε_HOMO)'},
        8: {'Property': '⟨R²⟩', 'Unit': 'a₀²', 'Description': 'Electronic spatial extent'},
        9: {'Property': 'zpve', 'Unit': 'eV', 'Description': 'Zero point vibrational energy'},
        10: {'Property': 'U₀', 'Unit': 'eV', 'Description': 'Internal energy at 0 K'},
        11: {'Property': 'U', 'Unit': 'eV', 'Description': 'Internal energy at 298.15 K'},
        12: {'Property': 'H', 'Unit': 'eV', 'Description': 'Enthalpy at 298.15 K'},
        13: {'Property': 'G', 'Unit': 'eV', 'Description': 'Free energy at 298.15 K'},
        14: {'Property': 'C_v', 'Unit': 'cal/molK', 'Description': 'Heat capacity at 298.15 K'}
    }

import torch

def get_atomic_property_tensors(energy_units="eV"):
    """
    Returns a dictionary of PyTorch tensors for each property,
    indexed by atomic number (Z), filled with 0.0 for elements not listed.
    """
    size = 10  # since max Z = 9 for F

    ZPVE = torch.zeros(size)
    U_0K = torch.zeros(size)
    U_298K = torch.zeros(size)
    H_298K = torch.zeros(size)
    G_298K = torch.zeros(size)
    CV = torch.zeros(size)

    # Fill values directly
    ZPVE[1] = ZPVE[6] = ZPVE[7] = ZPVE[8] = ZPVE[9] = 0.0

    U_0K[1] = -0.500273
    U_0K[6] = -37.846772
    U_0K[7] = -54.583861
    U_0K[8] = -75.064579
    U_0K[9] = -99.718730

    U_298K[1] = -0.498857
    U_298K[6] = -37.845355
    U_298K[7] = -54.582445
    U_298K[8] = -75.063163
    U_298K[9] = -99.717314

    H_298K[1] = -0.497912
    H_298K[6] = -37.844411
    H_298K[7] = -54.581501
    H_298K[8] = -75.062219
    H_298K[9] = -99.716370

    G_298K[1] = -0.510927
    G_298K[6] = -37.861317
    G_298K[7] = -54.598897
    G_298K[8] = -75.079532
    G_298K[9] = -99.733544

    CV[1] = CV[6] = CV[7] = CV[8] = CV[9] = 2.981

    if energy_units == "eV":
        conv_factor = energy_conversion("Ha","eV")
        ZPVE *= conv_factor
        U_0K *= conv_factor
        U_298K *= conv_factor
        H_298K *= conv_factor
        G_298K *= conv_factor

    return {
        "ZPVE": ZPVE,
        "U0": U_0K,
        "U": U_298K,
        "H": H_298K,
        "G": G_298K,
        "Cv": CV
    }


def load_qm9_dataset(
    folder_path,
    atom_properties=["zeff"],
    OH_node_features=[0],
    minmax_node_features=None,
    r_max=5.0,
    normalize=True,
    energy_units = "eV",
    return_stats=False,
    target = list(range(3,15))
):


    # Create node features dataframe
    properties_df = create_node_features(["H", "C", "N", "O", "F"], atom_properties)
    
    # Min max normalization
    if minmax_node_features:
        norm_columns = properties_df.iloc[:,minmax_node_features]
        properties_df.iloc[:,minmax_node_features] = (norm_columns-norm_columns.min())/(norm_columns.max()-norm_columns.min())

    # One Hot features
    if OH_node_features:
        OH_column_names = properties_df.columns[OH_node_features].tolist()
        properties_df = pd.get_dummies(properties_df, columns=OH_column_names, dtype=int)
        
    # Load dataset
    dataset = QM9Dataset(
        root=folder_path,
        properties_df=properties_df,
        r_max=r_max,
        transform=SetTarget(target=target),
    )
    print(f"Total number of samples: {len(dataset)}.")

    # One-hot encode atomic number
    dataset.data.z = F.one_hot(dataset.data.z.long()).squeeze().to(torch.float32)

    # Manage y
    y = dataset.y.view(len(dataset), -1)

    # Energy units
    conv_factor = energy_conversion("Ha", energy_units)
    energy_index = [*list(range(5,8)),*list(range(9,14))]
    y[:,energy_index] *= conv_factor
    dataset.data.y = y.flatten() 

    # Normalize target if requested
    if normalize:
        # Get mean and std
        std = y.std(dim=0).view(1,-1)
        mean = y.mean(dim=0).view(1,-1)
        atom_mean = mean.clone()
        atom_std = std.clone()

        # Dipole, polarizability and Spatial extent aren't shifted (otherwise norm doesn't behave correctly)
        mean[:,[3,4,8]] = 0   

        # Extensive properties mean and std are computed per atom
        atom_ref = get_atomic_property_tensors()
        batch = torch.repeat_interleave(torch.arange(len(dataset.n_atoms)), dataset.n_atoms)
        atom_ref = get_atomic_property_tensors()
        refs =  [atom_ref["ZPVE"][dataset.Z],        # ZPVE
                 atom_ref["U0"][dataset.Z],       # U0
                 atom_ref["U"][dataset.Z],       # U
                 atom_ref["H"][dataset.Z],       # H
                 atom_ref["G"][dataset.Z],       # G
                 atom_ref["Cv"][dataset.Z]]       # cv
        refs = torch.hstack(refs)
        refs = torch_scatter.scatter_sum(refs,index=batch,dim=0)
        atomization_values = y[:,9:15] - refs
        atomizatization_mean = atomization_values.mean(dim=0).view(1,-1)
        atomization_std = atomization_values.std(dim=0).view(1,-1)
        mean[:,9:15] = atomizatization_mean
        std[:,9:15] = atomization_std
        per_atom_values = atomization_values/dataset.n_atoms.view(-1,1).expand(-1,6)
        atom_mean_ = per_atom_values.mean(dim=0).view(1,-1)
        atom_std_ = per_atom_values.std(dim=0).view(1,-1)
        atom_mean[:,9:15] = atom_mean_
        atom_std[:,9:15] = atom_std_

    if return_stats:
        return dataset, mean.view(-1), std.view(-1), atom_mean.view(-1), atom_std.view(-1)
    else:
        return dataset 