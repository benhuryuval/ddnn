import argparse
import torch
import csv
from collections import defaultdict

from tqdm import tqdm

import datasets
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from grad_descent import get_optimal_alpha
from info_export import create_fxs_file, reorder_phis_file, adjust_sample_numbering
from model_settings import define_settings
from visual_tools import plot_snr_graphs, plot_coefficients

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

def test_model(model, test_loader, sigma, exit, alpha, dataset, export_info="no"):
    """
    Test the performance of the given model with specific noise covariance (sigma), exit strategy, and aggregation weights (alpha).

    Parameters:
    - model: PyTorch model to be evaluated.
    - test_loader: PyTorch DataLoader providing the test dataset.
    - sigma: Noise covariance matrix to be applied to the model during inference.
    - exit: Specifies which part of the model to use for early exit or final exit.
    - alpha: Aggregation weights used by the model for the evaluation (can be optimal, equal, or random depending on the method).
    - dataset: The name of the dataset to be used for generating the filename.
    - export_info: If "yes", collect info to create fxs and phis.

    Returns:
    - Accuracy (float): The accuracy of the model on the test dataset, expressed as a percentage.
    """

    model.eval()  # Set the model to evaluation mode
    num_correct = 0  # Counter for correct predictions

    # Iterate over the test data
    for data, target in tqdm(test_loader, leave=False):

        # Transfer data and target labels to the computation device
        data, target = data.to(), target.to()
        data, target = Variable(data), Variable(target)

        # Perform inference with the model using the given sigma, exit, and alpha
        predictions = model(data, sigma, exit, alpha, dataset, export_info)

        # Use the last prediction (output) for final classification
        last_pred = predictions[-1]  # Dimensions: [batch_size, num_classes], e.g., [32, 10]

        # Compute the loss (optional step; it's used for model evaluation but not returned)
        loss = F.cross_entropy(last_pred, target, size_average=False).item()

        # Get the predicted class with the highest probability
        pred = last_pred.data.max(1, keepdim=True)[1]  # Dimensions: [batch_size, 1]

        # Compare predictions to the ground truth and count the correct ones
        correct = (pred.view(-1) == target.view(-1)).long().sum().item()
        num_correct += correct  # Accumulate correct predictions

        # Export true labels to CSV if export_fxs is enabled
        if export_info == "yes":
            # Convert the true labels from the current batch to a DataFrame
            df_true_labels = pd.DataFrame({'true_label': target.cpu().numpy()})

            # Write the true labels to the CSV file in append mode
            df_true_labels.to_csv(f"fxs_{dataset}_true_label.csv", mode='a', header=not bool(num_correct), index=False)

    # Calculate and return the accuracy as a percentage
    N = len(test_loader.dataset)  # Total number of samples in the test dataset
    return 100. * (num_correct / N)



def evaluate_snr_performance(exit, alpha_agg_method, noise_cov_mat, out_filepath_csv, plot_graph="yes", snr_values=None):
    """
    Evaluate the performance of a model across different SNR values, using various aggregation strategies for alpha.
    The function writes the accuracy and normalized alpha values (if applicable) to separate CSV files.

    Parameters:
    - exit: str, Specifies which part of the model to use for early exit or final exit ('global' or 'local').
    - alpha_agg_method: str, Method for computing the alpha values. Can be 'optimal', 'equal', or 'random'.
    - noise_cov_mat: np.array, The noise covariance matrix to apply to the model during inference.
    - out_filepath_csv: str, The file path where the accuracy results will be stored. A separate file will be created for alpha values.
    - plot_graph: str, Indicates whether to plot the graph of the results ('yes' or 'no').
    - snr_values: list or None, Optional list of SNR values. If None, defaults to predefined values based on the dataset.

    Returns:
    - None. The function writes to CSV files and prints the model's accuracy at each SNR value.

    Usage Example:
    - evaluate_snr_performance(exit='local', alpha_agg_method='optimal',
                             noise_cov_mat=np.diag([1, 1, 1, 1, 1, 1]),
                             out_filepath_csv='local_mnist_optimal_noise1.csv')
    - evaluate_snr_performance(exit="local", alpha_agg_method="optimal",
                noise_cov_mat=np.diag([1, 1, 1, 10, 10, 10]), out_filepath_csv="local_cifar_optimal_noise2.csv")
    """
    # Define constants for cxx values
    cxx_global_mnist = 36.946195351075666
    cxx_local_mnist = 112.82436884928032
    cxx_local_cifar = 32.9801139938582

    alpha = []
    model, dataset, test_loader, train_loader, num_devices = define_settings()

    # default SNR values
    if snr_values is None:
        if dataset == "mnist":
            snr_values = [0.001, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
        elif dataset == "cifar10":
            snr_values = [0.001, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]

    # Create a CSV file for alphas
    alpha_filepath_csv = out_filepath_csv.replace(".csv", "_alphas.csv")

    if alpha_agg_method == "equal":
        alpha = np.ones(num_devices)  # Equal weights for all devices
    elif alpha_agg_method == "random":
        alpha = np.random.rand(num_devices)  # Random numbers between 0 and 1 for each device

    for index, snr in enumerate(snr_values):
        if exit == "global" and dataset == "mnist":
            cxx = cxx_global_mnist
        if exit == "local" and dataset == "mnist":
            cxx = cxx_local_mnist
        if exit == "local" and dataset == "cifar10":
            cxx = cxx_local_cifar

        sigma = noise_cov_mat * (cxx / (snr * np.trace(noise_cov_mat)))

        # Determine alpha based on aggregation strategy
        if alpha_agg_method == "optimal":
            alpha = get_optimal_alpha(sigma, dataset)  # Optimal alpha based on std
            # Normalize alphas
            normalized_alpha = alpha / np.sum(alpha)  # Normalize to sum to 1
            # Write normalized alpha to the CSV file
            with open(alpha_filepath_csv, mode='a') as f:
                f.write(','.join(map(str, normalized_alpha)) + '\n')


        acc = test_model(model, test_loader, sigma, exit, alpha, dataset)
        print('SNR = {:.4f}, ACC = {:.4f}'.format(snr, acc))
        data = [[snr, acc]]
        # Convert the matrix to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Write the DataFrame to a CSV file
        df.to_csv(out_filepath_csv, mode='a', index=False, header=False)

    # Make it a graph:
    if plot_graph == "yes":
        plot_snr_graphs(title="SNR vs ACC", filepath1=out_filepath_csv, label1="")


def simulate_random_alphas(exit, noise_cov_mat, compare_performance_csv, num_iterations, out_filepath_prefix):
    """
    Evaluate the performance of a model across different SNR values using random weights for alpha.
    The function saves the maximum accuracy, average accuracy, and the count of iterations exceeding
    the accuracy from a comparison CSV file for each SNR value.

    Parameters:
    - exit: str, Specifies which part of the model to use for early exit or final exit ('global' or 'local').
    - noise_cov_mat: np.array, The noise covariance matrix to apply to the model during inference.
    - compare_performance_csv: str, The file path of the CSV to compare results against.
    - num_iterations: int, The number of iterations to perform.
    - out_filepath_prefix: str, The prefix for output CSV files (without extension).

    Returns:
    - None. The function writes to CSV files containing the performance statistics.

    Usage Example:
    -     simulate_random_alphas(exit="local", noise_cov_mat=np.diag([1, 1, 1, 10, 10, 10]),
                           compare_performance_csv="local_cifar_optimal_noise2.csv",
                           num_iterations=100, out_filepath_prefix="local_cifar_random100_noise2")
    """
    # SNR values (same as in the original function)
    model, dataset, test_loader, train_loader, num_devices = define_settings()
    snr_values = []
    if dataset == "mnist":
        snr_values = [0.001, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
    elif dataset == "cifar10":
        snr_values = [0.001, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]

    max_accuracy = {snr: float('-inf') for snr in snr_values}
    accuracy_records = {snr: [] for snr in snr_values}
    exceed_count = {snr: 0 for snr in snr_values}

    # Load the comparison performance data
    comparison_data = pd.read_csv(compare_performance_csv, header=None)
    comparison_accuracies = dict(zip(comparison_data[0], comparison_data[1]))

    for i in range(num_iterations):
        print(f"Iteration: {i+1}")  # Print the current iteration number

        # Run evaluate_snr_performance with alpha_agg_method="random" and plot_graph="no"
        out_filepath_csv = f"{out_filepath_prefix}_{i}.csv"  # Unique CSV for each iteration
        evaluate_snr_performance(exit, "random", noise_cov_mat, out_filepath_csv, plot_graph="no")

        # Read the results from the output CSV file
        results = pd.read_csv(out_filepath_csv, header=None)
        for _, row in results.iterrows():
            snr, acc = row[0], row[1]

            # Update maximum accuracy
            if acc > max_accuracy[snr]:
                max_accuracy[snr] = acc

            # Store accuracy for averaging later
            accuracy_records[snr].append(acc)

        # After processing, delete the CSV file
        if os.path.exists(out_filepath_csv):
            os.remove(out_filepath_csv)

    # Calculate average accuracy and exceed counts
    average_accuracy = {snr: np.mean(accuracy_records[snr]) if accuracy_records[snr] else 0 for snr in snr_values}

    for snr in snr_values:
        # Count how many iterations exceeded the comparison accuracy
        if snr in comparison_accuracies:
            exceed_count[snr] = sum(acc > comparison_accuracies[snr] for acc in accuracy_records[snr])

    # Save the results to CSV files
    max_acc_filepath = out_filepath_prefix + "_max_accuracy.csv"
    avg_acc_filepath = out_filepath_prefix + "_average_accuracy.csv"
    exceed_count_filepath = out_filepath_prefix + "_exceed_count.csv"

    # Save max accuracy
    pd.DataFrame(max_accuracy.items(), columns=['SNR', 'Max Accuracy']).to_csv(max_acc_filepath, index=False, header=False)
    # Save average accuracy
    pd.DataFrame(average_accuracy.items(), columns=['SNR', 'Average Accuracy']).to_csv(avg_acc_filepath, index=False, header=False)
    # Save exceed count
    pd.DataFrame(exceed_count.items(), columns=['SNR', 'Exceed Count']).to_csv(exceed_count_filepath, index=False, header=False)

    print(f"Results saved to {max_acc_filepath}, {avg_acc_filepath}, and {exceed_count_filepath}.")

def create_phis_fxs():
    """
        Generates 'phis' and 'fxs' files for a given dataset and local exit.

        Parameters:
        - None

        Outputs:
        - 'phis' and 'fxs' files: Files containing processed model outputs for each device.

        Example Usage:
        - create_phis_fxs()
        """
    model, dataset, test_loader, train_loader, num_devices = define_settings()
    test_model(model, train_loader, sigma=np.diag([1, 1, 1, 1, 1, 1]), exit="local", alpha=np.ones(num_devices), dataset=dataset, export_info="yes")
    create_fxs_file(dataset)
    reorder_phis_file(dataset)

if __name__ == '__main__':
    model, dataset, test_loader, train_loader, num_devices = define_settings()



