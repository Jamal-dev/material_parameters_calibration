import pandas as pd
from pathlib import Path
from src.utility import row2Matrix
import numpy as np

def load_data(file_path:Path):
    """
    Load data from a CSV file.

    Args:
        file_path (Path): The path to the CSV file.

    Returns:
        DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(data: pd.DataFrame):
    """
    Preprocesses the given DataFrame by dropping rows with missing values.

    Args:
        data (pd.DataFrame): The input DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with missing values dropped.
    """
    data = data.dropna()
    return data


def extract_strain_stress(data:pd.DataFrame):
    """
    Extracts strain, stress, and loading direction from the given DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing strain, stress, and loading direction data.

    Returns:
        tuple: A tuple containing the following:
            - strain_lin (numpy.ndarray): An array of strain components in the order of 11, 22, 33, 12, 13, 23.
            - sigMatrixList (list): A list of stress matrices.
            - indices (dict): A dictionary containing indices for each strain component.
    """
    strain_lin = data.iloc[:,:6].values
    stress_lin = data.iloc[:,6:-1].values
    loading_direction = data['loading_cases'].values
    sigMatrixList = []
    for k in range(stress_lin.shape[0]):
        sigMatrixList.append(row2Matrix(stress_lin[k,:]))
    indices = {}
    indices['eps11'] = np.where(loading_direction == 1)[0]
    indices['eps22'] = np.where(loading_direction == 2)[0]
    indices['eps33'] = np.where(loading_direction == 3)[0]
    indices['eps12'] = np.where(loading_direction == 4)[0]
    indices['eps13'] = np.where(loading_direction == 5)[0]
    indices['eps23'] = np.where(loading_direction == 6)[0]
    return strain_lin, sigMatrixList, indices

def print_info(data: pd.DataFrame):
    """
    Prints information and summary statistics of the given DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame to print information and summary statistics for.

    Returns:
        None
    """
    print(data.info())
    print(data.describe())



def pipeline_preprocess(file_path: Path):
    """
    Preprocesses the data from the given file path.

    Args:
        file_path (Path): The path to the data file.

    Returns:
        tuple: A tuple containing the preprocessed strain, sigMatrixList, and indices.
    """
    data = load_data(file_path)
    data = preprocess_data(data)
    print_info(data)
    strain_lin, sigMatrixList, indices = extract_strain_stress(data)
    return strain_lin, sigMatrixList, indices

