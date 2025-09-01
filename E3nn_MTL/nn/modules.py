import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg

from typing import Dict, Union
from e3nn.nn.models.gate_points_2102 import Convolution, tp_path_exists
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import  Gate
import e3nn.o3 as o3
from e3nn.nn.models.gate_points_2101 import smooth_cutoff

from utils.graph_utils import get_atomic_property_tensors
from nn.out_blocks import Extensive, Intensive, Dipole_R2, Polarizability, Therm_Potential

class Graph_Preprocessing(nn.Module):
    def __init__(self, irreps_edge, max_radius, number_of_basis):
        super().__init__()
        self.irreps_edge = irreps_edge
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
    
    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)

        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_vec, True, normalization="component")
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None]*edge_sh

        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length, start=0.0, end=self.max_radius,
            number=self.number_of_basis, basis="gaussian", cutoff=False
        ).mul(self.number_of_basis ** 0.5)

        return edge_src, edge_dst, edge_attr, edge_length_embedded

class Equivariant_layer(nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_node_attr, irreps_edge,
                 max_radius, number_of_basis, radial_layers, radial_neurons, num_neighbors):
        super().__init__()
        self.irreps_in = irreps_in
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis

        # Number of edge features
        number_of_edge_features = number_of_basis + 2*irreps_node_attr.count("0e")

        # Activation functions
        act = {1: F.silu, -1: torch.tanh}               # For L=0 features
        act_gates = {1: torch.sigmoid, -1: torch.tanh}  # For scalars of L>0 features

        # L=0 features which use activation functions
        irreps_scalars = o3.Irreps([
            (mul, ir) for mul, ir in irreps_out
            if ir.l == 0 and tp_path_exists(irreps_in, irreps_edge, ir)
        ])

        # L>0 features which need a Gate
        irreps_gated = o3.Irreps([
            (mul, ir) for mul, ir in irreps_out
            if ir.l > 0 and tp_path_exists(irreps_in, irreps_edge, ir)
        ])
        ir = o3.Irreps("0e")[0][1] if tp_path_exists(irreps_in, irreps_edge, "0e") else o3.Irreps("0o")[0][1]

        # L=0 features for the Gates
        irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

        # Equivariant Activation layer
        self.gate = Gate(
            irreps_scalars,
            [act[ir.p] for _, ir in irreps_scalars],
            irreps_gates,
            [act_gates[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )
        self.irreps_out = self.gate.irreps_out

        # Equivariant Convolutional layer
        self.conv = Convolution(
            irreps_in = irreps_in,
            irreps_node_attr = irreps_node_attr,
            irreps_edge_attr = irreps_edge,
            irreps_out = self.gate.irreps_in,
            number_of_edge_features = number_of_edge_features,
            radial_layers = radial_layers,
            radial_neurons = radial_neurons,
            num_neighbors = num_neighbors,
        )

    def forward(self, x, z, edge_src, edge_dst, edge_attr, edge_scalars):
        x = self.conv(x, z, edge_src, edge_dst, edge_attr, edge_scalars)
        x = self.gate(x)
        return x

class Equivariant_module(nn.Module):
    def __init__(self, irreps_in, irreps_hidden, irreps_out, irreps_node_attr, irreps_edge, layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, num_neighbors, out_Gate=True):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis

        # Create the list of equivariant layers
        self.equivariant_layers = nn.ModuleList()
        irreps = irreps_in
        for i in range(layers-1):
            self.equivariant_layers.append(
                Equivariant_layer(
                    irreps_in=irreps,
                    irreps_out=irreps_hidden,
                    irreps_node_attr=irreps_node_attr,
                    irreps_edge=irreps_edge,
                    max_radius=max_radius,
                    number_of_basis=number_of_basis,
                    radial_layers=radial_layers,
                    radial_neurons=radial_neurons,
                    num_neighbors=num_neighbors
                )
            )
            irreps = self.equivariant_layers[-1].irreps_out
        
        # Output layer
        if out_Gate:
            self.equivariant_layers.append(
                Equivariant_layer(
                    irreps_in=irreps,
                    irreps_out=irreps_out,
                    irreps_node_attr=irreps_node_attr,
                    irreps_edge=irreps_edge,
                    max_radius=max_radius,
                    number_of_basis=number_of_basis,
                    radial_layers=radial_layers,
                    radial_neurons=radial_neurons,
                    num_neighbors=num_neighbors 
                )
            )
        else:
            self.equivariant_layers.append(
                Convolution(
                    irreps_in=irreps,
                    irreps_node_attr=irreps_node_attr,
                    irreps_edge_attr=irreps_edge,
                    irreps_out=irreps_out,
                    number_of_edge_features=number_of_basis + 2*irreps_node_attr.count("0e"),
                    radial_layers=radial_layers,
                    radial_neurons=radial_neurons,
                    num_neighbors=num_neighbors
                )
            )

        # Output irreps
        self.irreps_out = self.equivariant_layers[-1].irreps_out

    def forward(self, x, z, edge_src, edge_dst, edge_attr, edge_scalars):
        for layer in self.equivariant_layers:
            x = layer(x, z, edge_src, edge_dst, edge_attr, edge_scalars)
        return x

class Charges_module(nn.Module):
    def __init__(self, irreps_in, irreps_hidden, irreps_out, irreps_node_attr, irreps_edge, layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, num_neighbors,
                 mlp_neurons, mlp_layers):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out

        # Backbone for the charges, dipole and polarizability
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            out_Gate=True
        )

        # Blocks for {dipole, R2} and polarizability
        self.Dipole_R2 = Dipole_R2(scalars_in=irreps_out.count("0e"),mlp_neurons=mlp_neurons,mlp_layers=mlp_layers)
        self.polarizability = Polarizability(scalars_in=irreps_out.count("0e"),
                                             mlp_neurons=mlp_neurons, mlp_layers=mlp_layers)

    def forward(self, x, z, data, edge_src, edge_dst, edge_attr, edge_scalars):
        # Backbone
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)
        
        # Separate vector inputs
        in_scalars = x[:, :self.irreps_out.count("0e")]
        v1 = x[:, self.irreps_out.count("0e"):self.irreps_out.count("0e") + 3]
        v2 = x[:, self.irreps_out.count("0e") + 3:self.irreps_out.count("0e") + 6]
        in_dipoles = torch.cat([in_scalars,v1],dim=1)
        in_polarizability = torch.cat([in_scalars,v2],dim=1)

        # Calculate dipole, R2
        dipole, R2 = self.Dipole_R2(in_dipoles, data)

        # Polarizability
        polarizability = self.polarizability(in_polarizability, data)

        return dipole, polarizability, R2


class Energies_module(nn.Module):
    def __init__(self, irreps_in, irreps_hidden, irreps_out, irreps_node_attr, irreps_edge, layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, num_neighbors,
                 mlp_neurons, mlp_layers):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out

        # Backbone 
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            out_Gate=True
        )

        # Mean and std hardcoded for each target (HOMO, LUMO, gap)
        self.register_buffer("mean", torch.tensor([-6.530098, 0.30269313, 6.8327894]))
        self.register_buffer("std", torch.tensor([0.602227, 1.2771955, 1.2930548]))

        # HOMO and gap MLPs
        self.LUMO = Intensive(scalars_in=irreps_out.count("0e"), mlp_neurons=mlp_neurons, 
                                mlp_layers=mlp_layers, mean=self.mean[1], std=self.std[1])
        self.gap = Intensive(scalars_in=irreps_out.count("0e"), mlp_neurons=mlp_neurons, 
                                mlp_layers=mlp_layers, mean=self.mean[2], std=self.std[2])

    def forward(self, x, z, data, edge_src, edge_dst, edge_attr, edge_scalars):
        # Backbone
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)

        # LUMO and gap MLPs
        LUMO = self.LUMO(x,data)
        gap = self.gap(x,data)

        # HOMO is calculated from gap and LUMO
        HOMO = LUMO-gap

        # Output
        out = torch.cat([HOMO,LUMO,gap],dim=1)
        return out

class Thermodynamic_module(nn.Module):
    def __init__(self, irreps_in, irreps_hidden, irreps_out, irreps_node_attr, irreps_edge, layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, num_neighbors,
                 mlp_neurons, mlp_layers):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out

        # Mean and std of each target (ZPVE, U0, U, H, G , Cv)
        self.register_buffer("mean", torch.tensor([0.22239308, -4.2440658, -4.2699003, -4.293997, -3.9510736, -1.203925]))
        self.register_buffer("std", torch.tensor([0.015784923, 0.18961433, 0.18886884, 0.18869501, 0.18782417, 0.19224654]))

        # Common backbone for all extensive properties
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            out_Gate=True
        )

        # Atomic reference values for each target
        atom_ref = get_atomic_property_tensors()
        # Register each atomic reference as a buffer
        for k, v in atom_ref.items():
            self.register_buffer(f"atom_ref_{k}", v)


        # MLPs for each output
        self.out_blocks = nn.ModuleList()
        for i, prop in enumerate(["ZPVE", "U0", "U", "H", "G", "Cv"]):
            self.out_blocks.append(
                Extensive(scalars_in=irreps_out.count("0e"), mlp_neurons=mlp_neurons, 
                                mlp_layers=mlp_layers, mean=self.mean[i], std=self.std[i],
                                atom_ref=getattr(self, f"atom_ref_{prop}"))
            )

    def forward(self, x, z, data, edge_src, edge_dst, edge_attr, edge_scalars):
        # Backbone for all targets
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)

        # Outputs
        out = []
        for block in self.out_blocks:
            out.append(block(x, data))

        # Output
        out = torch.cat(out, dim=1)
        return out


class Charges_module_v2(nn.Module):
    def __init__(self, irreps_in, irreps_hidden, irreps_out, irreps_node_attr, irreps_edge, layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, num_neighbors,
                 mlp_neurons, mlp_layers):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out

        # Backbone for the charges, dipole and polarizability
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_hidden,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            out_Gate=True
        )

        # Equivariant block for each task
        self.charges_equivariant = Equivariant_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=1,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            out_Gate=True
        )

        self.polarizability_equivariant = Equivariant_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=1,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            out_Gate=True
        )


        # Blocks for {dipole, R2} and polarizability
        self.Dipole_R2 = Dipole_R2(scalars_in=irreps_out.count("0e"),mlp_neurons=mlp_neurons,mlp_layers=mlp_layers)
        self.polarizability = Polarizability(scalars_in=irreps_out.count("0e"),
                                             mlp_neurons=mlp_neurons, mlp_layers=mlp_layers)

    def forward(self, x, z, data, edge_src, edge_dst, edge_attr, edge_scalars):
        # Backbone
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)

        # Calculate dipole, R2
        x_dipole = self.charges_equivariant(x, z, edge_src, edge_dst, edge_attr, edge_scalars)
        dipole, R2 = self.Dipole_R2(x_dipole, data)

        # Polarizability
        x_polarizability = self.polarizability_equivariant(x, z, edge_src, edge_dst, edge_attr, edge_scalars)
        polarizability = self.polarizability(x_polarizability, data)

        return dipole, polarizability, R2


class Thermodynamic_module_v2(nn.Module):
    def __init__(self, irreps_in, irreps_hidden, irreps_out, irreps_node_attr, irreps_edge, layers,
                 max_radius, number_of_basis, radial_layers, radial_neurons, num_neighbors,
                 mlp_neurons, mlp_layers):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out

        # Mean and std of each target (ZPVE, U0, U, H, G , Cv)
        self.register_buffer("mean", torch.tensor([0.22239308, -4.2440658, -4.2699003, -4.293997, -3.9510736, -1.203925]))
        self.register_buffer("std", torch.tensor([0.015784923, 0.18961433, 0.18886884, 0.18869501, 0.18782417, 0.19224654]))

        # Atomic reference values for each target
        atom_ref = get_atomic_property_tensors()
        # Register each atomic reference as a buffer
        for k, v in atom_ref.items():
            self.register_buffer(f"atom_ref_{k}", v)

        # Common backbone for all extensive properties
        self.backbone = Equivariant_module(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_hidden,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
            out_Gate=True
        )

        # One equivariant block for ZPVE, other for c_v and other for thermodynamic potentials
        self.zpve_equivariant = Equivariant_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=1,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors
        )

        self.cv_equivariant = Equivariant_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge = irreps_edge,
            layers=1,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors
        )

        self.Potential_backbone = Equivariant_module(
            irreps_in=self.backbone.irreps_out,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge=irreps_edge,
            layers=1,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            radial_layers=radial_layers,
            radial_neurons=radial_neurons,
            num_neighbors=num_neighbors,
        )

        # Output blocks
        self.ZPVE = Extensive(scalars_in=irreps_out.count("0e"), mlp_neurons=mlp_neurons, 
                                mlp_layers=mlp_layers, mean=self.mean[0], std=self.std[0],
                                atom_ref=atom_ref["ZPVE"])
        
        self.therm_potentials = nn.ModuleList()
        for i, prop in enumerate(["U0", "U", "H", "G"]):
            self.therm_potentials.append(
                Extensive(scalars_in=irreps_out.count("0e"), mlp_neurons=mlp_neurons, 
                                mlp_layers=mlp_layers, mean=self.mean[i+1], std=self.std[i+1],
                                atom_ref=atom_ref[prop]))

        self.cv = Extensive(scalars_in=irreps_out.count("0e"), mlp_neurons=mlp_neurons, 
                        mlp_layers=mlp_layers, mean=self.mean[5], std=self.std[5],
                        atom_ref=atom_ref["Cv"])

    def forward(self, x, z, data, edge_src, edge_dst, edge_attr, edge_scalars):
        # Backbone for all targets
        x = self.backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)

        # Outputs
        out = []

        # ZPVE 
        x_ZPVE = self.zpve_equivariant(x, z, edge_src, edge_dst, edge_attr, edge_scalars)
        out.append(self.ZPVE(x_ZPVE, data))

        # Thermodynamic potentials 
        x_pot = self.Potential_backbone(x, z, edge_src, edge_dst, edge_attr, edge_scalars)
        for therm_potential in self.therm_potentials:
            out.append(therm_potential(x_pot, data))
        
        # CV
        x_cv = self.cv_equivariant(x, z, edge_src, edge_dst, edge_attr, edge_scalars)
        out.append(self.cv(x_cv, data))

        # Output
        out = torch.cat(out, dim=1)
        return out
