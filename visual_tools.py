import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_snr_graphs(title, filepath1, label1, filepath2="", label2="", filepath3="", label3="", filepath4="", label4="", skip_rows=0):
    """
        Plot SNR (Signal-to-Noise Ratio) vs Accuracy graphs from multiple CSV files on the same graph.

        Parameters:
        - title: str, the title of the plot.
        - filepath1: str, path to the first CSV file.
        - label1: str, the label for the first dataset in the plot.
        - skip_rows: int, number of rows to skip in the CSV file (optional, default is 0).

        Returns:
        - None.

        Usage Example:
        -     plot_snr_graphs(
        "Local Exit Accuracy Performances with Noise Covariance Matrix Σ2 on CIFAR10 Dataset Using Different Aggregation Methods",
        "local_cifar_equal_noise2.csv", "equal weights (average)",
        "local_cifar_optimal_noise2.csv", "optimal weights",
        "local_cifar_random100_noise2_max_accuracy.csv",
        "random weights - maximum (out of 100 trials)",
        "local_cifar_random100_noise2_average_accuracy.csv",
        "random weights - average (over 100 trials)")
    """
    plt.figure(figsize=(10, 6))

    data1 = pd.read_csv(filepath1, header=None, names=['snr', 'acc'], skiprows=skip_rows)
    snr1, acc1 = data1['snr'], data1['acc']
    plt.plot(snr1, acc1, marker='x', linestyle='-', color='g', label=label1)

    if filepath2 != "":
        data2 = pd.read_csv(filepath2, header=None, names=['snr', 'acc'], skiprows=skip_rows)
        snr2, acc2 = data2['snr'], data2['acc']
        plt.plot(snr2, acc2, marker='x', linestyle='-', color='r', label=label2)

    if filepath3 != "":
        data3 = pd.read_csv(filepath3, header=None, names=['snr', 'acc'], skiprows=skip_rows)
        snr3, acc3 = data3['snr'], data3['acc']
        plt.plot(snr3, acc3, marker='x', linestyle='-', color='#800080', label=label3)

    if filepath4 != "":
        data4 = pd.read_csv(filepath4, header=None, names=['snr', 'acc'], skiprows=skip_rows)
        snr4, acc4 = data4['snr'], data4['acc']
        plt.plot(snr4, acc4, marker='x', linestyle='-', color='b', label=label4)

    # Add title and labels
    plt.title(title)
    plt.xlabel('SNR')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Add legend
    plt.legend()

    # Set the number of ticks
    num_ticks_x = len(snr1)
    num_ticks_y = 11

    # Generate evenly spaced ticks for x-axis and y-axis
    x_ticks = np.linspace(np.floor(snr1[0]), np.floor(snr1[num_ticks_x-1]), num_ticks_x)
    y_ticks = np.linspace(0, 100, num_ticks_y)

    # Set the ticks on x-axis and y-axis
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # Show the plot
    plt.show()


def plot_coefficients(csv_file_path, bar_width=0.2):
    """
    Plots a comparison of coefficient values under different noise conditions
    (high noise, middle noise, weak noise) for sub-regressor indices.

    Parameters:
    - csv_file_path: str, Path to the CSV file containing coefficient values for different noise conditions.
    - bar_width: float, The width of the bars for each noise condition. Default is 0.2.

    Returns:
    - None, displays the plot.

    Usage Example:
    - plot_coefficients("local_mnist_optimal_noise2_snr_zero_inf_alphas.csv")
    """

    # Load the CSV file
    df = pd.read_csv(csv_file_path, header=None)

    # Extract the coefficient values for each noise condition
    high_noise = df.iloc[0, :].values
    mid_noise = df.iloc[1, :].values
    weak_noise = df.iloc[2, :].values

    # Y-axis: Sub-regressor indices (1 to 6)
    indices = np.arange(1, len(weak_noise) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot bars for each noise level
    plt.barh(indices + bar_width, high_noise, height=bar_width, label='High Noise', color='purple', edgecolor='black')
    plt.barh(indices, mid_noise, height=bar_width, label='Mid Noise', color='steelblue', edgecolor='black')
    plt.barh(indices - bar_width, weak_noise, height=bar_width, label='Weak Noise', color='green', edgecolor='black')

    # Add labels and title
    plt.xlabel('Coefficient Value')
    plt.ylabel('Sub Regressor Index')
    plt.yticks(indices, labels=indices)
    plt.title('Comparison of Coefficient Values Under Different Noise Conditions')

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Show plot
    plt.show()


def histogram(filename_csv, bins, xlabel, ylabel, title):
    """
    Generate a histogram from the specified column of a CSV file and calculate the average of the values.

    Parameters:
    - filename_csv: str, Path to the CSV file containing the data.
    - bins: int, Number of bins to use for the histogram.
    - xlabel: str, Label for the x-axis.
    - ylabel: str, Label for the y-axis.
    - title: str, Title of the histogram.

    Returns:
    - None. The function displays a histogram and writes the average to a text file.

    Usage Example:
    - histogram(filename_csv='trc_local_cifar.csv', bins=100, xlabel='Trace Value',
     ylabel='Number of Features', title='Histogram of Covariance Matrix Trace Values')
    """
    # Read the data from the CSV file
    df = pd.read_csv(filename_csv, header=None)

    # Extract the data from the second column (index 1)
    traces = df.iloc[:, 1]  # Adjust this if you want a different column

    # Remove NaN values if any
    traces = traces.dropna()

    # Plot the histogram
    plt.hist(traces, bins=bins)  # Use the parameter for bins
    plt.xlabel(xlabel)            # Use the parameter for x-axis label
    plt.ylabel(ylabel)            # Use the parameter for y-axis label
    plt.title(title)              # Use the parameter for title
    plt.show()

    # Calculate the average using pandas
    average = traces.mean()

    # Sanity check
    num_rows = len(traces)
    print("Number of rows extracted:", num_rows)
    print("Average of the trace of covariance features matrix:", average)

    # Write average to a text file with a dynamic filename
    output_filename = filename_csv.replace('.csv', '_average_trace.txt')
    with open(output_filename, "w") as f:
        f.write(f"Average of the traces: {average}\n")







