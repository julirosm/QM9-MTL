import torch, sys, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        if hasattr(self, "grad_index"):
            pass
        else:
            self.grad_index = []
            for param in self.get_share_params():
                self.grad_index.append(param.data.numel())
            self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode):
        '''
        mode: backward, autograd
        '''
        grads = torch.zeros(self.out_channels, self.grad_dim).to(self.device)
        for tn in range(self.out_channels):
            if mode == 'backward':
                losses[tn].backward(retain_graph=True) if (tn+1)!=self.out_channels else losses[tn].backward()
                grads[tn] = self._grad2vec()
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                grads[tn] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError('No support {} mode for gradient computation')
            self.zero_grad_share_params()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass


class DB_MTL(AbsWeighting):

    def __init__(self, DB_beta=0.9, DB_beta_sigma=0.5):
        nn.Module.__init__(self)
        self.beta = DB_beta
        self.beta_sigma = DB_beta_sigma

    def init_param(self):
        self.step = 0
        self._compute_grad_dim()
        self.grad_buffer = torch.zeros(self.out_channels, self.grad_dim).to(self.device)
        
    def backward(self, losses):
        self.step += 1

        batch_weight = np.ones(len(losses))
        self._compute_grad_dim()
        batch_grads = self._compute_grad(torch.log(losses+1e-8), mode='backward') # [out_channels, grad_dim]

        self.grad_buffer = batch_grads + (self.beta/self.step**self.beta_sigma) * (self.grad_buffer - batch_grads)

        u_grad = self.grad_buffer.norm(dim=-1)

        alpha = u_grad.max() / (u_grad + 1e-8)
        new_grads = sum([alpha[i] * self.grad_buffer[i] for i in range(self.out_channels)])

        self._reset_grad(new_grads)
        return batch_weight