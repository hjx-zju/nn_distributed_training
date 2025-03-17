import numpy as np

def remove_outliers(data, threshold=3):
    """
    Remove outliers from the data using Z-score method.

    Args:
        data (np.ndarray): Input data array.
        threshold (float): Z-score threshold to identify outliers.

    Returns:
        np.ndarray: Data with outliers removed.
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    filtered_data = data[z_scores < threshold]
    return filtered_data

def calculate_mean(data, threshold=3):
    """
    Calculate the mean of the data after removing outliers.

    Args:
        data (np.ndarray): Input data array.
        threshold (float): Z-score threshold to identify outliers.

    Returns:
        float: Mean of the data with outliers removed.
    """
    filtered_data = remove_outliers(data, threshold)
    return np.mean(filtered_data)
