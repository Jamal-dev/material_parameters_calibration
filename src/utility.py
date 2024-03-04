import numpy as np
import torch

def row2Matrix(row):
    """
    Converts a row of values into a 3x3 matrix.

    Parameters:
    row (list): A list of 6 values representing the elements of the row.

    Returns:
    numpy.ndarray: A 3x3 matrix created from the row values.
    """
    mat = np.array([[row[0], row[3], row[4]], \
                    [row[3], row[1], row[5]], \
                    [row[4], row[5], row[2]]])
    return mat

def estimate_F_C(b, J):
    """
    Estimates the deformation gradient F and the right Cauchy-Green deformation tensor C
    given the left Cauchy-Green deformation tensor b and the volume change J.

    Parameters:
    b (numpy.ndarray): The left Cauchy-Green deformation tensor.
    J (float): The volume change.

    Returns:
    F (numpy.ndarray): The deformation gradient.
    C (numpy.ndarray): The right Cauchy-Green deformation tensor.
    error (float): The error between the estimated b and the actual b.
    """
    # Perform spectral decomposition of b
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    
    # Since J = det(F), and for isotropic materials det(F) = det(v),
    # adjust the principal stretches to reflect volume change
    principal_stretches = np.sqrt(eigenvalues) * np.cbrt(J / np.prod(np.sqrt(eigenvalues)))

    # Reconstruct F using the principal stretches and the eigenvectors of b
    # Note: This reconstruction assumes F can be directly derived from b's eigendecomposition, 
    # which may not capture the full rotation effects in v. This is a simplification.
    U = np.diag(principal_stretches)
    F = eigenvectors @ U @ np.linalg.inv(eigenvectors)

    # Calculate C = F.T * F
    C = np.matmul(F.T, F)
    b_estimate = np.matmul(F, F.T)
    error = np.linalg.norm(b - b_estimate)

    return F, C, error


def doubleDotProduct(A, B):
    """
    Calculates the double dot product of two matrices A and B.

    Parameters:
    A (numpy.ndarray): The first matrix.
    B (numpy.ndarray): The second matrix.

    Returns:
    float: The result of the double dot product.
    """
    return np.trace(np.matmul(A.T, B))


def deviatoric(sig):
    """
    Compute the deviatoric part of a symmetric tensor.

    Parameters:
        sig (numpy.ndarray): The input symmetric tensor.

    Returns:
        numpy.ndarray: The deviatoric part of the input tensor.
    """
    return sig - 1/3*np.trace(sig)*np.eye(3)


class Bases:
    """
    A class representing bases.

    Attributes:
        base1 (torch.Tensor): The first base.
        base2 (torch.Tensor): The second base.
        base3 (torch.Tensor): The third base.
        base4 (torch.Tensor): The fourth base.
        base5 (torch.Tensor): The fifth base.
    """

    def __init__(self, base1, base2, base3, base4, base5):
        self.base1 = torch.tensor(base1, dtype=torch.float32)
        self.base2 = torch.tensor(base2, dtype=torch.float32)
        self.base3 = torch.tensor(base3, dtype=torch.float32)
        self.base4 = torch.tensor(base4, dtype=torch.float32)
        self.base5 = torch.tensor(base5, dtype=torch.float32)

    def to_dict(self):
        """
        Convert the bases to a dictionary.

        Returns:
            dict: A dictionary representation of the bases.
        """
        return {
            'base1': self.base1,
            'base2': self.base2,
            'base3': self.base3,
            'base4': self.base4,
            'base5': self.base5,
        }

    def __repr__(self) -> str:
        text = f'''base1: {self.base1}, 
                   base2: {self.base2}, 
                   base3: {self.base3},
                   base4: {self.base4},
                   base5: {self.base5}'''
        return text

def split_train_test(dataset, train_size=0.88, val_size=0.91):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    dataset (MyDataset): The dataset to split.
    train_size (float): The proportion of the dataset to include in the training set.
    val_size (float): The proportion of the dataset to include in the validation set.

    Returns:
    tuple: A tuple containing the following datasets:
        - train_dataset (MyDataset): The training dataset.
        - val_dataset (MyDataset): The validation dataset.
        - test_dataset (MyDataset): The test dataset.
    """
    dataset_size = len(dataset)
    train_split = int(train_size * dataset_size)
    val_split = int(val_size * dataset_size)
    indices = torch.randperm(dataset_size).tolist()
    # split indices
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    # create datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, val_dataset, test_dataset


import re

def validate_expression(expression):
    """
    Validates an expression to ensure it only contains allowed variables and constants.

    Args:
        expression (str): The expression to be validated.

    Raises:
        ValueError: If an unauthorized variable or constant is found in the expression.

    Returns:
        None
    """
    # Allowed variables and pattern for constants (c1, c2, ..., cn)
    allowed_variables = ['J', 'I1bar', 'I2bar', 'I4bar', 'I5bar']
    constant_pattern = r'\bc\d+\b'  # Pattern to match 'c' followed by digits
    
    # Find all variables and constants in the expression
    variables_constants = re.findall(r'[A-Za-z]\w*', expression)
    
    # Check for unauthorized variables or constants
    for item in variables_constants:
        if item not in allowed_variables and not re.match(constant_pattern, item):
            # use only allowed variables
            msg = f"Unauthorized variable or constant '{item}' found in expression." \
                     + f"Use only {allowed_variables} and constants c1, c2, ..., cn"
            raise ValueError(msg)
    
    # Check if the expression uses only the allowed variables and constants
    for var in allowed_variables:
        if var in expression:
            continue
        else:
            # Optionally, raise an error if any allowed variable is missing
            # raise ValueError(f"Required variable '{var}' is missing from the expression.")
            pass

    # If the expression passes all checks
    print("Expression is valid.")


def modify_expression(expression):
    expression = expression.replace('J', 'X[:,2]')
    expression = expression.replace('I1bar', 'X[:,0]')
    expression = expression.replace('I2bar', 'X[:,1]')
    expression = expression.replace('I4bar', 'X[:,3]')
    expression = expression.replace('I5bar', 'X[:,4]')
    return expression

def cauchy_stress_vectorized(X,dPsidX,base1,base2,base3,base4,base5):
    """
    Calculates the Cauchy stress tensor for a given set of input variables.

    Args:
        X (torch.Tensor): Input tensor containing the values of I1_bar, I2_bar, J, I4_bar, I5_bar.
        dPsidX (torch.Tensor): Input tensor containing the derivatives of the potential function with respect to I1_bar, I2_bar, J, I4_bar, I5_bar.
        base1 (torch.Tensor): Base tensor 1.
        base2 (torch.Tensor): Base tensor 2.
        base3 (torch.Tensor): Base tensor 3.
        base4 (torch.Tensor): Base tensor 4.
        base5 (torch.Tensor): Base tensor 5.

    Returns:
        torch.Tensor: Cauchy stress tensor.

    """
    dPsidI1bar = dPsidX[:,0]
    dPsidI2bar = dPsidX[:,1]
    dPsidJ = dPsidX[:,2]
    dPsidI4bar = dPsidX[:,3]
    dPsidI5bar = dPsidX[:,4]
    I1bar = X[:,0]
    dPsidI1bar_expanded = dPsidI1bar.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
    dPsidI2bar_expanded = dPsidI2bar.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
    dPsidJ_expanded = dPsidJ.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
    dPsidI4bar_expanded = dPsidI4bar.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
    dPsidI5bar_expanded = dPsidI5bar.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
    I1bar_expanded = I1bar.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
    sig = (dPsidI1bar_expanded + I1bar_expanded * dPsidI2bar_expanded) * base1 + \
            + dPsidI2bar_expanded * base2 +  \
            + dPsidJ_expanded * base3 + \
            + dPsidI4bar_expanded * base4 + \
            + dPsidI5bar_expanded * base5
    return sig