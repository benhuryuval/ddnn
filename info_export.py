import os

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



def create_fxs_file(dataset):
    """
    Create a merged CSV file with predicted class probabilities, true labels, and predicted class.

    Parameters:
    - dataset: str, the name of the dataset

    Returns:
    - None
    """


    # Define the filenames for probabilities and true labels
    prob_filename = f"fxs_{dataset}_prob.csv"
    label_filename = f"fxs_{dataset}_true_label.csv"

    # Define column names for probabilities (assuming 10 classes) and true labels
    prob_columns = [str(i) for i in range(10)]  # Class probabilities columns
    label_columns = ['true_labels']  # True labels column

    # Read probabilities and true labels from separate CSV files, adding headers
    prob_df = pd.read_csv(prob_filename, header=None, names=prob_columns)
    label_df = pd.read_csv(label_filename, header=None, names=label_columns)

    # Verify that the dataframes match in the number of samples
    assert prob_df.shape[0] == label_df.shape[0], "Mismatch in the number of samples between probabilities and true labels."

    # Determine the predicted class by taking the index of the maximum probability for each sample
    predicted_class = np.argmax(prob_df.values, axis=1)

    # Create the merged DataFrame
    merged_df = pd.concat([prob_df, label_df, pd.DataFrame(predicted_class, columns=['f(x)'])], axis=1)

    # Create the filename for the merged CSV
    merged_filename = f"fxs_{dataset}.csv"

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(merged_filename, index=False)
    print(f"Merged file saved as {merged_filename}")

    # Delete the temporary CSV files used for the merge
    os.remove(prob_filename)
    os.remove(label_filename)
    print(f"Deleted temporary files: {prob_filename}, {label_filename}")


def reorder_phis_file(dataset):
    """
    Reorder the phis_<dataset>.csv file according to specific rules and add headers.

    Parameters:
    - dataset: str, the name of the dataset (used for the filename).

    Returns:
    - None
    """
    # Create the filename for the CSV file
    filename = f"phis_{dataset}.csv"

    # Read the existing CSV file without headers
    df = pd.read_csv(filename, header=None)

    # Assign column names
    df.columns = ['Sample', 'Device', 'Class', 'Prediction']

    # Initialize a new column to hold the updated sample index
    new_samples = []

    # Calculate new sample indices based on batch counting
    for index, row in df.iterrows():
        # Determine the batch number (0-based index)
        batch_number = index // 1920

        # Update the sample number
        new_sample_idx = row['Sample'] + (32 * batch_number)
        new_samples.append(new_sample_idx)

    # Add the new samples to the DataFrame
    df['New_Sample'] = new_samples

    # Sort the DataFrame based on the new sample index, device, and class
    sorted_df = df.sort_values(by=['New_Sample', 'Device', 'Class']).reset_index(drop=True)

    # Drop the old 'Sample' column and rename 'New_Sample' to 'Sample'
    sorted_df.drop(columns=['Sample'], inplace=True)
    sorted_df.rename(columns={'New_Sample': 'Sample'}, inplace=True)

    # Reorder the columns to ensure they are in the correct order
    sorted_df = sorted_df[['Sample', 'Device', 'Class', 'Prediction']]

    # Save the reordered DataFrame back to the same CSV file with headers
    sorted_df.to_csv(filename, index=False, header=True)
    print(f"Reordered file saved as {filename}")




def adjust_sample_numbering(dataset):
    """
    Adjust the sample numbering in the phis_<dataset>.csv file based on batch logic.

    Parameters:
    - dataset: str, the name of the dataset (used for the filename).

    Returns:
    - None
    """
    # Create the filename for the CSV file
    filename = f"phis_{dataset}.csv"

    # Read the existing CSV file without headers
    df = pd.read_csv(filename, header=None)

    # Assign column names (maintaining the original column order)
    df.columns = ['Sample', 'Device', 'Class', 'Prediction']

    # Initialize the batch number and a sample number counter
    batch_number = 0
    samples_per_batch = 32

    # Iterate through each row to calculate the new sample index
    for index, row in df.iterrows():
        # Check for the start of a new batch based on (0, 0, 0)
        if row['Device'] == 0 and row['Class'] == 0 and row['Sample'] == 0:
            # Increment the batch number
            batch_number += 1

        # Calculate the new sample index
        new_sample_idx = (batch_number - 1) * samples_per_batch + index % samples_per_batch
        df.at[index, 'Sample'] = new_sample_idx  # Update the sample number in the DataFrame

    # Save the updated DataFrame back to the same CSV file without headers
    df.to_csv(filename, index=False, header=False)
    print(f"Sample numbering adjusted and saved as {filename}")

