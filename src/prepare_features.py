from src.utility import Bases, deviatoric
from src.utility import estimate_F_C, doubleDotProduct, row2Matrix
import numpy as np
import torch

def calc_bases(bbar,F,J,direction):
    """
    Calculate the bases for a given set of parameters.

    Parameters:
    - bbar: ndarray, shape (3, 3)
        The input bbar matrix.
    - F: ndarray, shape (3, 3)
        The input F matrix.
    - J: float
        The input J value.
    - direction: ndarray, shape (3,)
        The input direction vector.

    Returns:
    - bases: Bases
        The calculated bases.

    """
    dev_bbar = deviatoric(bbar)
    base1 = 2.0/J * dev_bbar
    base2 = -2.0/J * deviatoric(bbar @ bbar)
    Fbar = J**(-1/3)*F
    a_bar = Fbar @ direction
    a_bar_dash = bbar @ a_bar
    base3 = np.eye(3) 
    base4 = 2.0 * a_bar @ a_bar.T
    base5 = 2.0 * a_bar @ a_bar_dash.T + 2.0 * a_bar_dash @ a_bar.T
    return Bases(base1,base2,base3,base4,base5)

def creating_invariances(eps_nominal_lin:np.ndarray, fiber_direction:np.ndarray):
    """
    Calculate invariances based on the given input arrays.

    Args:
        eps_nominal_lin (np.ndarray): Array of nominal strains.
        fiber_direction (np.ndarray): Array of fiber directions.

    Returns:
        tuple: A tuple containing the following arrays:
            - v_mat_lin (np.ndarray): Array of v_mat values.
            - b_mat_lin (np.ndarray): Array of b_mat values.
            - invarices (np.ndarray): Array of invariances.
            - invarices_bar (np.ndarray): Array of invariances_bar.
            - bases_data (list): List of base objects.
    """
    v_mat_lin = np.zeros((len(eps_nominal_lin),6))
    b_mat_lin = np.zeros((len(eps_nominal_lin),6))
    I1 = np.zeros((len(eps_nominal_lin),1))
    I2 = np.zeros((len(eps_nominal_lin),1))
    I3 = np.zeros((len(eps_nominal_lin),1))
    I4 = np.zeros((len(eps_nominal_lin),1))
    I5 = np.zeros((len(eps_nominal_lin),1))
    I1_bar = np.zeros((len(eps_nominal_lin),1))
    I2_bar = np.zeros((len(eps_nominal_lin),1))
    I4_bar = np.zeros((len(eps_nominal_lin),1))
    I5_bar = np.zeros((len(eps_nominal_lin),1))
    fiber_dyad = np.matmul(fiber_direction,fiber_direction.T)
    bases_data = []

    for k in range(eps_nominal_lin.shape[0]):
        eps_nom_matrix = row2Matrix(eps_nominal_lin[k,:])
        v_mat = eps_nom_matrix + np.eye(3)
        b_mat = np.matmul(v_mat,v_mat)
        J = np.linalg.det(v_mat)
        F, C,error = estimate_F_C(b_mat,J)
        bbar = (J)**(-2/3)*b_mat
        base_obj = calc_bases(bbar,F,J,fiber_direction)
        Cbar = (J)**(-2/3)*C
        # Calculate invariants
        I1[k] = np.trace(b_mat)
        I2[k] = 0.5*(I1[k]**2 - np.trace(np.matmul(b_mat,b_mat)))
        I3[k] = J
        I4[k] = doubleDotProduct(C,fiber_dyad)
        I5[k] = doubleDotProduct(C @ C,fiber_dyad)
        
        
        I1_bar[k] = np.trace(bbar)
        I2_bar[k] = 0.5*(I1_bar[k]**2 - np.trace(np.matmul(bbar,bbar)))
        I4_bar[k] = doubleDotProduct(Cbar,fiber_dyad)
        I5_bar[k] = doubleDotProduct(Cbar @ Cbar,fiber_dyad)
        
        bases_data.append(base_obj) 
        v_mat_lin[k,:] = [v_mat[0,0],v_mat[1,1],v_mat[2,2], \
                            v_mat[0,1],v_mat[0,2],v_mat[1,2]]
        b_mat_lin[k,:] = [b_mat[0,0],b_mat[1,1],b_mat[2,2], \
                            b_mat[0,1],b_mat[0,2],b_mat[1,2]]
    invarices = np.concatenate((I1,I2,I3,I4,I5),axis=1)
    invarices_bar = np.concatenate((I1_bar,I2_bar,I3,I4_bar,I5_bar),axis=1)
    return v_mat_lin,b_mat_lin,invarices,invarices_bar,bases_data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # convert invariants to tensor
        invariants = torch.tensor(data['invriants'], dtype=torch.float32)
        invariants.requires_grad = True
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
    
    def __getitem__(self, index):
        # Extract data for a single sample
        invariants = self.invariants[index]
        base1 = self.base1_list[index,:,:]
        base2 = self.base2_list[index,:,:]
        base3 = self.base3_list[index,:,:]
        base4 = self.base4_list[index,:,:]
        base5 = self.base5_list[index,:,:]
        bases = (base1, base2, base3, base4, base5)
        # target_stress = self.target_stress[index]
        target_stress = self.target_stress[index,:,:]
        return invariants, bases, target_stress

    def __len__(self):
        return len(self.invariants)