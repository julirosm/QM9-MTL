from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg

from e3nn.nn import ExtractIr
import e3nn.o3 as o3

from utils.graph_utils import get_atomic_property_tensors
from nn.modules import Graph_Preprocessing, Equivariant_module, Charges_module, Energies_module, Thermodynamic_module, Charges_module_v2, Thermodynamic_module_v2
from nn.out_blocks import Dipole, Polarizability, Intensive, R2, Extensive, Dipole_R2


class Multitasking(torch.nn.Module):
    def __init__(self, x_dim, z_dim, emb_dim, l_max, common_layers, branch_layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, 
                 mlp_layers, mlp_neurons):
        super().__init__()

        self.irreps_edge = o3.Irreps.spherical_harmonics(1)
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.out_channels = 12
        self.emb_dim = emb_dim
        
        # Determine hidden irreps
        parity = {-1:"o",1:"e"}
        irreps_hidden = ""
        for i in range(l_max+1):
            p = parity[(-1)**i]
            irreps_hidden += f" {emb_dim}x{i}{p} "
            if i is not l_max:
                irreps_hidden += "+"
        self.irreps_hidden = o3.Irreps(irreps_hidden)

        # Irreps out
        self.irreps_out = o3.Irreps(f"{emb_dim}x0e")
        
        
        # Preprocessing of data
        self.graph_preprocessing = Graph_Preprocessing(self.irreps_edge, max_radius, number_of_basis)

        # Embedding of data.x, data.z
        self.x_emb = nn.Linear(x_dim, emb_dim)
        self.z_emb = nn.Linear(z_dim, emb_dim)

        # Define node attributes
        irreps_in = o3.Irreps(str(emb_dim) + "x0e")
        irreps_node_attr = o3.Irreps(str(emb_dim) + "x0e")
        self.ext_z = ExtractIr(str(emb_dim) + "x0e", "0e")

        # Backbone of the network
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_hidden,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=common_layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            out_Gate=True

        )

        # Dipole, R2 and polarizability
        self.charges_module = Charges_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_out=o3.Irreps(f"{emb_dim}x0e + 2x1o"),
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=branch_layers[0],
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            mlp_neurons=mlp_neurons, 
            mlp_layers=mlp_layers,
        )

        # HOMO, LUMO and gap
        self.energies_module = Energies_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=branch_layers[1],
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            mlp_neurons=mlp_neurons, 
            mlp_layers=mlp_layers
        )

        # ZPVE, U0, U, G, H and cv
        self.thermodynamic_module = Thermodynamic_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=branch_layers[2],
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            mlp_neurons=mlp_neurons, 
            mlp_layers=mlp_layers
        )

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:

        # Preprocess the graph data
        edge_src, edge_dst, edge_attr, edge_length_embedded = self.graph_preprocessing(data)

        # Embed the node features and atomic numbers
        z = self.z_emb(data.z.float())
        scalar_z = self.ext_z(z)
        edge_scalars = torch.cat([edge_length_embedded, scalar_z[edge_src], scalar_z[edge_dst]], dim=1)

        x = self.x_emb(data.x)

        # Common backbone for all targets
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)

        # Process the different targets
        mu, alpha, R2 = self.charges_module(x, z, data, edge_src, edge_dst, edge_attr, edge_scalars)
        out_energies = self.energies_module(x, z, data, edge_src, edge_dst, edge_attr, edge_scalars)
        out_thermodynamic = self.thermodynamic_module(x, z, data, edge_src, edge_dst, edge_attr, edge_scalars)

        # Concatenate the outputs
        out = torch.cat([mu, alpha, out_energies, R2, out_thermodynamic], dim=1)

        return out
    
class Singletasking(torch.nn.Module):
    def __init__(self, x_dim, z_dim, emb_dim, target, l_max, layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, 
                 mlp_layers, mlp_neurons):
        super().__init__()

        self.irreps_edge = o3.Irreps.spherical_harmonics(1)
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.out_channels = 1
        self.emb_dim = emb_dim
        self.register_buffer("mean", torch.tensor([0.0,0.0, 0.0, -6.530098, 0.30269313, 6.8327894, 0.0, 0.22239308, -4.2440658, -4.2699003, -4.293997, -3.9510736, -1.203925]))
        self.register_buffer("std", torch.tensor([1.0, 1.0, 1.0, 0.602227, 1.2771955, 1.2930548, 1.0, 0.015784923, 0.18961433, 0.18886884, 0.18869501, 0.18782417, 0.19224654]))
        
        # Determine hidden irreps
        parity = {-1:"o",1:"e"}
        irreps_hidden = ""
        for i in range(l_max+1):
            p = parity[(-1)**i]
            irreps_hidden += f" {emb_dim}x{i}{p} "
            if i is not l_max:
                irreps_hidden += "+"
        self.irreps_hidden = o3.Irreps(irreps_hidden)

        # Irreps out
        self.irreps_out = o3.Irreps(f"{emb_dim}x0e")
        
        
        # Preprocessing of data
        self.graph_preprocessing = Graph_Preprocessing(self.irreps_edge, max_radius, number_of_basis)

        # Embedding of data.x, data.z
        self.x_emb = nn.Linear(x_dim, emb_dim)
        self.z_emb = nn.Linear(z_dim, emb_dim)

        # Define node attributes
        irreps_in = o3.Irreps(str(emb_dim) + "x0e")
        irreps_node_attr = o3.Irreps(str(emb_dim) + "x0e")
        self.ext_z = ExtractIr(str(emb_dim) + "x0e", "0e")

        # Define output representation
        if target in [3,4]:
            self.irreps_out = o3.Irreps(f"{emb_dim}x0e + 1x1o")
        else:
            self.irreps_out = o3.Irreps(f"{emb_dim}x0e")

        # Backbone of the network
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            out_Gate=True
        )

        # Ouput of the network
        if target == 3:
            self.out = Dipole(
                scalars_in=emb_dim,
                mlp_neurons=mlp_neurons,
                mlp_layers=mlp_layers,
                atomic_dipoles=True,
                correct_charges=True
            )
        elif target == 4:
            self.out = Polarizability(
                scalars_in=emb_dim,
                mlp_neurons=mlp_neurons,
                mlp_layers=mlp_layers
            )
        elif target in [5,6,7]:
            self.out = Intensive(
                scalars_in=emb_dim,
                mlp_neurons=mlp_neurons,
                mlp_layers=mlp_layers,
                mean=self.mean[target-3],
                std=self.std[target-3]
            )
        elif target == 8:
            self.out = R2(
                scalars_in=emb_dim,
                mlp_neurons=mlp_neurons,
                mlp_layers=mlp_layers
            )
        elif target in [9,10,11,12,13,14]:
            atom_ref = get_atomic_property_tensors()
            prop_dict = {9:"ZPVE", 10:"U0", 11:"U", 12:"H", 13:"G", 14:"Cv"}
            for k, v in atom_ref.items():
                self.register_buffer(f"atom_ref_{k}", v)
            self.out = Extensive(
                scalars_in=emb_dim,
                mlp_neurons=mlp_neurons,
                mlp_layers=mlp_layers,
                mean=self.mean[target-3],
                std=self.std[target-3],
                atom_ref=getattr(self, f"atom_ref_{prop_dict[target]}")
            )
        else:
            self.out = nn.Identity()
            print("Warning: No output block defined, using identity.")


    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:

        # Preprocess the graph data
        edge_src, edge_dst, edge_attr, edge_length_embedded = self.graph_preprocessing(data)

        # Embed the node features and atomic numbers
        z = self.z_emb(data.z.float())
        scalar_z = self.ext_z(z)
        edge_scalars = torch.cat([edge_length_embedded, scalar_z[edge_src], scalar_z[edge_dst]], dim=1)

        x = self.x_emb(data.x)

        # Common backbone for all targets
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)

        # Concatenate the outputs
        out = self.out(x, data)
        return out


class Multitasking_v2(torch.nn.Module):
    def __init__(self, x_dim, z_dim, emb_dim, l_max, common_layers, branch_layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, 
                 mlp_layers, mlp_neurons):
        super().__init__()

        self.irreps_edge = o3.Irreps.spherical_harmonics(1)
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.out_channels = 12
        self.emb_dim = emb_dim
        
        # Determine hidden irreps
        parity = {-1:"o",1:"e"}
        irreps_hidden = ""
        for i in range(l_max+1):
            p = parity[(-1)**i]
            irreps_hidden += f" {emb_dim}x{i}{p} "
            if i is not l_max:
                irreps_hidden += "+"
        self.irreps_hidden = o3.Irreps(irreps_hidden)

        # Irreps out
        self.irreps_out = o3.Irreps(f"{emb_dim}x0e")
        
        
        # Preprocessing of data
        self.graph_preprocessing = Graph_Preprocessing(self.irreps_edge, max_radius, number_of_basis)

        # Embedding of data.x, data.z
        self.x_emb = nn.Linear(x_dim, emb_dim)
        self.z_emb = nn.Linear(z_dim, emb_dim)

        # Define node attributes
        irreps_in = o3.Irreps(str(emb_dim) + "x0e")
        irreps_node_attr = o3.Irreps(str(emb_dim) + "x0e")
        self.ext_z = ExtractIr(str(emb_dim) + "x0e", "0e")

        # Backbone of the network
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_hidden,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=common_layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            out_Gate=True

        )

        # Dipole, R2 and polarizability
        self.charges_module = Charges_module_v2(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_out=o3.Irreps(f"{emb_dim}x0e + 1x1o"),
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=branch_layers[0],
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            mlp_neurons=mlp_neurons, 
            mlp_layers=mlp_layers,
        )

        # HOMO, LUMO and gap
        self.energies_module = Energies_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=branch_layers[1],
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            mlp_neurons=mlp_neurons, 
            mlp_layers=mlp_layers
        )

        # ZPVE, U0, U, G, H and cv
        self.thermodynamic_module = Thermodynamic_module_v2(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=branch_layers[2],
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            mlp_neurons=mlp_neurons, 
            mlp_layers=mlp_layers
        )

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:

        # Preprocess the graph data
        edge_src, edge_dst, edge_attr, edge_length_embedded = self.graph_preprocessing(data)

        # Embed the node features and atomic numbers
        z = self.z_emb(data.z.float())
        scalar_z = self.ext_z(z)
        edge_scalars = torch.cat([edge_length_embedded, scalar_z[edge_src], scalar_z[edge_dst]], dim=1)

        x = self.x_emb(data.x)

        # Common backbone for all targets
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)

        # Process the different targets
        mu, alpha, R2 = self.charges_module(x, z, data, edge_src, edge_dst, edge_attr, edge_scalars)
        out_energies = self.energies_module(x, z, data, edge_src, edge_dst, edge_attr, edge_scalars)
        out_thermodynamic = self.thermodynamic_module(x, z, data, edge_src, edge_dst, edge_attr, edge_scalars)

        # Concatenate the outputs
        out = torch.cat([mu, alpha, out_energies, R2, out_thermodynamic], dim=1)

        return out


class Charges_model(torch.nn.Module):
    def __init__(self, x_dim, z_dim, emb_dim, l_max, layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, 
                 mlp_layers, mlp_neurons):
        super().__init__()

        self.irreps_edge = o3.Irreps.spherical_harmonics(1)
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.out_channels = 2
        self.emb_dim = emb_dim

        # Irreps hidden
        parity = {-1:"o",1:"e"}
        irreps_hidden = ""
        for i in range(l_max+1):
            p = parity[(-1)**i]
            irreps_hidden += f" {emb_dim}x{i}{p} "
            if i is not l_max:
                irreps_hidden += "+"
        self.irreps_hidden = o3.Irreps(irreps_hidden)

        # Irreps out
        self.irreps_out = o3.Irreps(f"{emb_dim}x0e + 1x1o")
        
        
        # Preprocessing of data
        self.graph_preprocessing = Graph_Preprocessing(self.irreps_edge, max_radius, number_of_basis)

        # Embedding of data.x, data.z
        self.x_emb = nn.Linear(x_dim, emb_dim)
        self.z_emb = nn.Linear(z_dim, emb_dim)

        # Define node attributes
        irreps_in = o3.Irreps(str(emb_dim) + "x0e")
        irreps_node_attr = o3.Irreps(str(emb_dim) + "x0e")
        self.ext_z = ExtractIr(str(emb_dim) + "x0e", "0e")

        # Backbone of the network
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=self.irreps_edge,
            layers=layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=16.5397,
            out_Gate=True
        )
        
        self.out = Dipole_R2(emb_dim,mlp_neurons, mlp_layers)

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        # Preprocess the graph data
        edge_src, edge_dst, edge_attr, edge_length_embedded = self.graph_preprocessing(data)

        # Embed the node features and atomic numbers
        z = self.z_emb(data.z.float())
        scalar_z = self.ext_z(z)
        edge_scalars = torch.cat([edge_length_embedded, scalar_z[edge_src], scalar_z[edge_dst]], dim=1)

        x = self.x_emb(data.x)

        # Common backbone for all targets
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)

        # Process the different targets
        mu, R2 = self.out(x, data)

        out = torch.cat([mu, R2], dim=1)

        return out
    
    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.parameters()
    
    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.zero_grad(set_to_none=False)