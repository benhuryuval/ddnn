import numpy as np
import pandas as pd

def cov_of_feature(i, j, k, X):
    """
    Calculate the covariance of a specific feature (i, j, k) across 6 devices in the matrix X, and store the trace of this covariance matrix in a CSV file.

    Parameters:
    - i: int, The index for the first feature dimension in X.
    - j: int, The index for the second feature dimension in X.
    - k: int, The index for the third feature dimension in X.
    - X: np.array, The input matrix of shape (N, features, height, width). The matrix contains features from multiple devices, with the features of each device concatenated along the second axis.

    Returns:
    - None. The function writes the trace of the covariance matrix for the feature (i, j, k) to a CSV file.

    Process:
    - Extracts feature (i, j, k) for all devices.
    - Constructs a matrix where each row corresponds to a sample, and each column corresponds to a device's feature.
    - Computes the covariance matrix (Cov_ijk) of the feature across devices.
    - Calculates the trace of this covariance matrix.
    - Appends the trace along with feature indices (i, j, k) to the 'trc.csv' file.
    """

    N = X.shape[0]  # Number of samples
    dim = int(X.shape[1] / 6)  # 16 - which is the length to "jump" between devices in the matrix X

    # Initialize a matrix to store the values for the current feature across devices
    matrix = np.zeros((N, 6))

    # Populate the matrix by extracting the (i, j, k) feature for each device
    for r in range(N):
        for t in range(6):
            matrix[r][t] = X[r][i + dim * t][j][k]

    # Calculate the covariance matrix of the extracted feature across devices
    cov = (matrix.T @ matrix) / N

    # Compute the trace of the covariance matrix
    trc = np.trace(cov)

    # Prepare the data for writing to CSV
    data = [[i, j, k, trc]]
    df = pd.DataFrame(data)

    # Append the result to the CSV file
    csv_file_path = 'trc.csv'
    df.to_csv(csv_file_path, mode='a', index=False, header=False)


def export_predictions_to_csv(all_predictions, filename):
    df = pd.DataFrame(all_predictions)
    df.to_csv(filename, mode='a', index=False, header=False)

