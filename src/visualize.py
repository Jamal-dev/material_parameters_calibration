import matplotlib.pyplot as plt
import torch
import numpy as np
from src.utility import cauchy_stress_vectorized

def plot_losses(train_losses, val_losses):
    """
    Plot the training and validation losses.

    Parameters:
    train_losses (list): A list of training losses.
    val_losses (list): A list of validation losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



class Visualization:
    """
    A class for visualizing stress and strain data.

    Args:
        data (dict): A dictionary containing the data for visualization.
        indices (dict): A dictionary containing indices for each strain component.

    Attributes:
        invariants (torch.Tensor): Tensor of invariants.
        bases (list): List of bases.
        base1_list (torch.Tensor): Tensor of base1 values.
        base2_list (torch.Tensor): Tensor of base2 values.
        base3_list (torch.Tensor): Tensor of base3 values.
        base4_list (torch.Tensor): Tensor of base4 values.
        base5_list (torch.Tensor): Tensor of base5 values.
        target_stress (torch.Tensor): Tensor of target stress values.
        plot_configs (list): List of plot configurations.
        fontsize (int): Font size for the plots.
    """
    def __init__(self, data, indices):
        # convert invariants to tensor
        invariants = torch.tensor(data['invriants'], dtype=torch.float32)
        invariants.requires_grad = True
        self.indices = indices
        self.invariants = invariants
        self.bases = data['bases']
        self.base1_list = [b.base1 for b in self.bases]
        self.base2_list = [b.base2 for b in self.bases]
        self.base3_list = [b.base3 for b in self.bases]
        self.base4_list = [b.base4 for b in self.bases]
        self.base5_list = [b.base5 for b in self.bases]
        self.base1_list = torch.stack(self.base1_list)
        self.base2_list = torch.stack(self.base2_list)
        self.base3_list = torch.stack(self.base3_list)
        self.base4_list = torch.stack(self.base4_list)
        self.base5_list = torch.stack(self.base5_list)
        # target_stress = torch.tensor(data['target_stress'], dtype=torch.float32)
        # target_stress.requires_grad = True
        target_stress = torch.stack(data['target_stress'])
        self.target_stress = target_stress
        self.plot_configs = [
            {'xrange': indices['eps11']  , 'eps_idx': 0, 'subplot_idx': 1, 'xy_idx':11},
            {'xrange': indices['eps22'], 'eps_idx': 1, 'subplot_idx': 2, 'xy_idx':22},
            {'xrange': indices['eps33'], 'eps_idx': 2, 'subplot_idx': 3, 'xy_idx':33},
            {'xrange': indices['eps12'], 'eps_idx': 3, 'subplot_idx': 4, 'xy_idx':12},
            {'xrange': indices['eps13'], 'eps_idx': 4, 'subplot_idx': 5, 'xy_idx':13},
            {'xrange': indices['eps23'], 'eps_idx': 5, 'subplot_idx': 6, 'xy_idx':23},
        ]
        self.fontsize = 14
    def forward(self,model):
        """
        Perform forward pass through the model.

        Args:
            model: The model to perform forward pass on.

        Returns:
            torch.Tensor: The predicted stress tensor.
        """
        psi = model(self.invariants)
        dPsidX = torch.autograd.grad(psi, self.invariants, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        sigma = cauchy_stress_vectorized(self.invariants,dPsidX,self.base1_list,self.base2_list,self.base3_list,self.base4_list,self.base5_list)
        return sigma
    def get_sigma_lin(self,model):
        """
        Get the predicted and labeled stress tensors in linear form.

        Args:
            model: The model to get the stress tensors from.

        Returns:
            tuple: A tuple containing the predicted stress tensor and the labeled stress tensor.
        """
        sigma = self.forward(model).detach().numpy()
        sigma_labeled = self.target_stress.detach().numpy()
        sigma_pred_lin = np.zeros((sigma.shape[0],6))
        sigma_labeled_lin = np.zeros((sigma.shape[0],6))
        for i in range(sigma.shape[0]):
            sigma_pred_lin[i,:] = [sigma[i,0,0],sigma[i,1,1],sigma[i,2,2],sigma[i,0,1],sigma[i,0,2],sigma[i,1,2]]
            sigma_labeled_lin[i,:] = [sigma_labeled[i,0,0],sigma_labeled[i,1,1],sigma_labeled[i,2,2],sigma_labeled[i,0,1],sigma_labeled[i,0,2],sigma_labeled[i,1,2]]
        return sigma_pred_lin,sigma_labeled_lin
    def get_psi(self,model):
        """
        Get the predicted strain energy.

        Args:
            model: The model to get the strain energy from.

        Returns:
            numpy.ndarray: The predicted strain energy.
        """
        model.eval()
        with torch.no_grad():
            psi = model(self.invariants)
        return psi.detach().numpy()
    def plot_stresses(self,model,epsMatrix ):
        """
        Plot the stress values.

        Args:
            model: The model to get the stress values from.
            epsMatrix: The matrix of strain values.
        """
        sig_list_pred, stressList_exp = self.get_sigma_lin(model)
        # Configuration settings
        plt.figure(figsize=(10, 15))  # Adjust figure size as needed
        for config in self.plot_configs:
            temp_range = config['xrange']
            eps_idx = config['eps_idx']
            subplot_idx = config['subplot_idx']
            
            eps = epsMatrix[temp_range, eps_idx]
            sig = sig_list_pred[temp_range, eps_idx]
            sig_exp = stressList_exp[temp_range, eps_idx]

            plt.subplot(3, 2, subplot_idx)
            plt.plot(eps, sig, label='Predicted')
            plt.plot(eps, sig_exp, '.', label='Experimental')
            plt.xlabel(f'$\epsilon_{{{config["xy_idx"]}}}$', fontsize=self.fontsize)
            plt.ylabel(f'$\sigma_{{{config["xy_idx"]}}}$', fontsize=self.fontsize)
            plt.legend()

        plt.tight_layout()
        plt.show()
    def plot_psi(self,model,epsMatrix ):
        """
        Plot the strain energy values.

        Args:
            model: The model to get the strain energy values from.
            epsMatrix: The matrix of strain values.
        """
        psi = self.get_psi(model)
        plt.figure(figsize=(10, 15))
        for config in self.plot_configs:
            xrange= config['xrange']
            eps_idx = config['eps_idx']
            subplot_idx = config['subplot_idx']

            eps = epsMatrix[xrange, eps_idx]

            plt.subplot(3, 2, subplot_idx)
            plt.plot(eps, psi[xrange])
            plt.xlabel(f'$\epsilon_{{{config["xy_idx"]}}}$', fontsize=self.fontsize)
            plt.ylabel(f'$\Psi$', fontsize=self.fontsize)
        plt.tight_layout()
        plt.show()