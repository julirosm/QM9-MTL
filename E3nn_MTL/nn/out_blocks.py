import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.SiLU, return_dummy=False, dummy_dim=3):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation())
        self.network = nn.Sequential(*layers)
        self.return_dummy = return_dummy
        self.dummy_dim = dummy_dim

    def forward(self, x):
        out = self.network(x)
        if self.return_dummy:
            dummy = torch.zeros(x.size(0), self.dummy_dim, device=x.device, dtype=x.dtype)
            return out, dummy
        return out


class Gated_block(nn.Module):
    '''
    Right now, just one vector input (The idea of this layer is being an output layer)
    '''
    def __init__(self, scalars_in, mlp_neurons, mlp_layers, activation=nn.SiLU):
        super().__init__()
        self.scalars_in = scalars_in

        self.mlp = MLP([scalars_in+1]+ (mlp_layers-1)*[mlp_neurons] + [2],activation)
    
    def forward(self, x):
        scalars = x[:,0:self.scalars_in]
        vectors = x[:,self.scalars_in:]
        x = torch.cat([scalars,torch.norm(vectors,p=2,dim=1).view(-1,1)],dim=1)
        x = self.mlp(x)
        scalars_out = x[:,0]
        vectors_out = x[:,1].view(-1,1)*vectors
        return scalars_out.view(-1,1), vectors_out


class Extensive(nn.Module):
    def __init__(self,scalars_in, mlp_neurons, mlp_layers, mean, std, atom_ref, activation = nn.SiLU):
        super().__init__()
        self.scalars_in = scalars_in
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("atom_ref", atom_ref)

        self.mlp = MLP([scalars_in]+ (mlp_layers-1)*[mlp_neurons] + [1])
    
    def forward(self, x, data):
        x = self.atom_ref[data.Z] + self.mean + self.std*self.mlp(x)
        x = torch_scatter.scatter_sum(x, data.batch, dim=0) 
        return x

class Therm_Potential(nn.Module):
    def __init__(self,scalars_in, mlp_neurons, mlp_layers, mean, std, atom_ref, activation = nn.SiLU):
        super().__init__()
        self.scalars_in = scalars_in
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("atom_ref", atom_ref)

        self.mlp = MLP([1]+ (mlp_layers-1)*[mlp_neurons] + [1])
    
    def forward(self, x, data):
        x = self.atom_ref[data.Z] + self.mean + self.std*(x+self.mlp(x))
        x = torch_scatter.scatter_sum(x, data.batch, dim=0) 
        return x

class Intensive(nn.Module):
    def __init__(self,scalars_in, mlp_neurons, mlp_layers, mean, std, activation = nn.SiLU):
        super().__init__()
        self.scalars_in = scalars_in
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

        self.mlp = MLP([scalars_in]+ (mlp_layers-1)*[mlp_neurons] + [1])
    
    def forward(self, x, data):  
        x = torch_scatter.scatter_mean(x, data.batch, dim=0) 
        x = self.mlp(x)*self.std + self.mean
        return x


class Dipole(nn.Module):
    def __init__(self, scalars_in, mlp_neurons, mlp_layers, activation=nn.SiLU,
                 atomic_dipoles=True, correct_charges=True):
        super().__init__()
        self.scalars_in = scalars_in
        self.atomic_dipoles = atomic_dipoles
        self.correct_charges = correct_charges
        if self.correct_charges:
            self.register_buffer("mean", torch.tensor(0.7546106515883616))
        else:
            self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(0.30338715545464656))
        self.register_buffer("eA_to_Debye", torch.tensor(4.802456245061274))
        self.register_buffer("atomic_masses", torch.tensor([0, 1.00784, 0, 0, 0, 0, 12.0107, 14.0067, 15.999, 18.998403]))

        if atomic_dipoles:
            self.out = Gated_block(scalars_in, mlp_neurons, mlp_layers)
        else:
            self.out = MLP([scalars_in]+ (mlp_layers-1)*[mlp_neurons] + [1], 
                           activation=activation, 
                           return_dummy=True, 
                           dummy_dim=3)
    
    def forward(self, x, data, return_norm=True):
        # Calculate molecules CM
        masses = self.atomic_masses[data.Z]
        CM = torch_scatter.scatter_sum(data.pos*masses,data.batch,dim=0)/torch_scatter.scatter_sum(masses, data.batch, dim=0)

        # Calculate charges and dipoles
        charges, atomic_dipoles = self.out(x)
        charges = charges*self.std + self.mean  # Denormalize charges

        # Make Q=0
        if self.correct_charges:
            total_charge = torch_scatter.scatter_mean(charges, data.batch, dim=0)
            charges = charges - total_charge[data.batch]

        # Calculate dipole
        dipole = torch_scatter.scatter_sum(charges*(data.pos-CM[data.batch])+atomic_dipoles, data.batch, dim=0)
        dipole = dipole*self.eA_to_Debye  # Convert to Debye

        # Calculate norm
        if return_norm:
            dipole = torch.norm(dipole,p=2,dim=1).view(-1,1)

        return dipole
    

class R2(nn.Module):
    def __init__(self,scalars_in, mlp_neurons, mlp_layers, activation = nn.SiLU, correct_charges=True):
        super().__init__()
        self.scalars_in = scalars_in
        self.correct_charges = correct_charges
        if self.correct_charges:
            self.register_buffer("mean", torch.tensor(0.7546106515883616))
        else:
            self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(0.30338715545464656))
        self.register_buffer("A_to_a0", torch.tensor(1.8897268777743552))
        self.register_buffer("atomic_masses", torch.tensor([0, 1.00784, 0, 0, 0, 0, 12.0107, 14.0067, 15.999, 18.998403]))


        self.out = MLP([scalars_in]+ (mlp_layers-1)*[mlp_neurons] + [1], 
                        activation=activation, 
                        return_dummy=False)
    
    def forward(self, x, data):
        # Calculate molecules CM
        masses = self.atomic_masses[data.Z]
        CM = torch_scatter.scatter_sum(data.pos*masses,data.batch,dim=0)/torch_scatter.scatter_sum(masses, data.batch, dim=0)

        charges = self.out(x)
        charges = charges*self.std + self.mean  # Denormalize charges
        if self.correct_charges:
            total_charge = torch_scatter.scatter_mean(charges, data.batch, dim=0)
            charges = charges - total_charge[data.batch]
        clouds = torch.abs((charges - data.Z))
        R2 = (clouds*torch.norm((data.pos-CM[data.batch])*self.A_to_a0,dim=1,keepdim=True)**2).view(-1,1)
        R2 = torch_scatter.scatter_sum(R2, data.batch, dim=0)

        return R2


class Dipole_R2(nn.Module):
    def __init__(self,scalars_in, mlp_neurons, mlp_layers, activation = nn.SiLU,
                 atomic_dipoles=True, correct_charges=True):
        super().__init__()
        self.scalars_in = scalars_in
        self.atomic_dipoles = atomic_dipoles
        self.correct_charges = correct_charges
        if self.correct_charges:
            self.register_buffer("mean", torch.tensor(0.7546106515883616))
        else:
            self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(0.30338715545464656))
        self.register_buffer("eA_to_Debye", torch.tensor(4.802456245061274))
        self.register_buffer("A_to_a0", torch.tensor(1.8897268777743552))
        self.register_buffer("atomic_masses", torch.tensor([0, 1.00784, 0, 0, 0, 0, 12.0107, 14.0067, 15.999, 18.998403]))

        if atomic_dipoles == True:
            self.out = Gated_block(scalars_in, mlp_neurons, mlp_layers)
        else:
            self.out = MLP([scalars_in]+ (mlp_layers-1)*[mlp_neurons] + [1], 
                           activation=activation, 
                           return_dummy=True, 
                           dummy_dim=3)
    
    def forward(self, x, data, return_norm=True):
        # Calculate molecules CM
        masses = self.atomic_masses[data.Z]
        CM = torch_scatter.scatter_sum(data.pos*masses,data.batch,dim=0)/torch_scatter.scatter_sum(masses, data.batch, dim=0)

        charges, atomic_dipoles = self.out(x)
        charges = charges*self.std + self.mean  # Denormalize charges
        if self.correct_charges:
            total_charge = torch_scatter.scatter_mean(charges, data.batch, dim=0)
            charges = charges - total_charge[data.batch]
        dipole = torch_scatter.scatter_sum(charges*(data.pos-CM[data.batch])+atomic_dipoles, data.batch, dim=0)
        dipole = dipole*self.eA_to_Debye  # Convert to Debye
        if return_norm:
            dipole = torch.norm(dipole,p=2,dim=1).view(-1,1)

        clouds = torch.abs((charges - data.Z))
        R2 = (clouds*torch.norm((data.pos-CM[data.batch])*self.A_to_a0,dim=1,keepdim=True)**2).view(-1,1)
        R2 = torch_scatter.scatter_sum(R2, data.batch, dim=0)

        return dipole, R2
    
class Polarizability(nn.Module):
    def __init__(self,scalars_in, mlp_neurons, mlp_layers, activation = nn.SiLU):
        super().__init__()
        self.scalars_in = scalars_in
        self.property_type = "Polarizability"
        self.register_buffer("std", torch.tensor(8.187783))
        self.register_buffer("atomic_masses", torch.tensor([0, 1.00784, 0, 0, 0, 0, 12.0107, 14.0067, 15.999, 18.998403]))


        self.out = Gated_block(scalars_in, mlp_neurons, mlp_layers, activation)
    
    def forward(self, x, data, return_norm=True):
        # Calculate molecules CM
        masses = self.atomic_masses[data.Z]
        CM = torch_scatter.scatter_sum(data.pos*masses,data.batch,dim=0)/torch_scatter.scatter_sum(masses, data.batch, dim=0)

        alpha, nu = self.out(x)
        alpha = alpha.expand(-1,3)
        alpha = torch.diag_embed(alpha)
        pos = data.pos-CM[data.batch]
        beta = nu.unsqueeze(1).expand(-1,3,3)*pos.unsqueeze(2).expand(-1,3,3)
        beta = beta + beta.transpose(1,2)
        polarizability = torch_scatter.scatter_sum(alpha+beta, data.batch, dim=0)*self.std
        if return_norm:
            polarizability = torch.linalg.matrix_norm(polarizability,ord='fro',dim=(1,2)).view(-1,1)
        return polarizability