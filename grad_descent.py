import numpy as np
import pandas as pd
from scipy.stats import norm

def load_fxs(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract the 12th column (11th index)
    column_data = df.iloc[:, 11].tolist()

    # Convert data to integers
    column_data = list(map(int, column_data))

    return column_data

def load_phis(file_path):
    """
    Load the 4th column from a CSV file and organize the data into a list of 6x10 matrices.

    Parameters:
    - file_path: str, the path to the CSV file.

    Returns:
    - A list of numpy arrays, each representing a 6x10 matrix.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract the 4th column (3rd index)
    column_data = df.iloc[:, 3].tolist()

    # Convert data to floats
    column_data = list(map(float, column_data))

    # Determine the number of matrices
    num_matrices = len(column_data) // 60

    # Create list of matrices
    matrices = []
    for i in range(num_matrices):
        # Extract 60 values for each matrix
        matrix_data = column_data[i * 60:(i + 1) * 60]

        # Reshape into 6x10 matrix and append to the list
        matrix = np.array(matrix_data).reshape(6, 10)
        matrices.append(matrix)

    return matrices


# Example function for Δ_(i,k)(x)
def delta_ik(phi, i, k):
    # Ensure the indices are within the valid range
    if i < 0 or i >= phi.shape[1] or k < 0 or k >= phi.shape[1]:
        raise IndexError("Column indices are out of bounds")

    # Subtract the k-th column from the i-th column
    result = phi[:, i] - phi[:, k]

    # Convert the result to a column vector
    result = result.reshape(-1, 1)

    return result

def grad_function(alpha, phis, f_xs, sigma, M):
    N = len(phis)
    grad = np.zeros_like(alpha)  # Initialize gradient as a 1D array
    for k in range(M):
        for f_x, phi in zip(f_xs, phis):
            if f_x == k:
                for i in range(M):  # M is the number of labels
                    if i != k:
                        delta = delta_ik(phi, i, k)
                        grad += internal_expression(alpha, sigma, delta).ravel()  # Ensure result is 1D

    grad *= 1 / (np.sqrt(2 * np.pi) * N)
    return grad


def internal_expression(alpha, sigma, delta):
    inside_sqrt = 2 * alpha.T @ sigma @ alpha
    z = (alpha.T @ delta) / np.sqrt(inside_sqrt)
    exp = np.exp(-0.5 * z ** 2)
    term1 = delta.flatten() / np.sqrt(inside_sqrt)  # Ensure term1 is 1D
    term2 = (alpha.T @ delta) * (sigma @ alpha) / (inside_sqrt ** 1.5)
    return (exp * (term1 - term2)).flatten()  # Return as a 1D array


def cost_function(alpha, phis, f_xs, sigma, M):
    N = len(phis)
    cost = 0
    for k in range(M):
        for f_x, phi in zip(f_xs, phis):
            if f_x == k:
                for i in range(M):  # M is the number of labels
                    if i != k:
                        delta = delta_ik(phi, i, k)
                        inside_sqrt = 2 * alpha.T @ sigma @ alpha
                        z = (alpha.T @ delta) / np.sqrt(inside_sqrt)
                        cost += norm.cdf(z)

    cost /= N
    return cost


# Gradient Descent Implementation
def gradient_descent(alpha_init, grad_fun, cost_fun, max_iter=30000, min_iter=10, tol=1e-5, learn_rate=0.2):
    """
        Perform gradient descent with AdaGrad-style learning rate adjustment.

        Parameters:
        - alpha_init: numpy array, the initial weight vector.
        - grad_fun: function, the function to compute the gradient.
        - cost_fun: function, the function to compute the cost.
        - max_iter: int, the maximum number of iterations.
        - min_iter: int, the minimum number of iterations before checking convergence.
        - tol: float, the tolerance for convergence.
        - learn_rate: float, the initial learning rate.

        Returns:
        - cost_evolution: list, the evolution of the cost over iterations.
        - alpha_evolution: list, the evolution of the weight vector over iterations.
        - i: int, the iteration at which the process stopped.
    """

    # initializations
    cost_evolution = [None] * max_iter
    alpha_evolution = [None] * max_iter
    grad_evolution = [None] * max_iter
    eps = 1e-15  # tolerance value for adagrad learning rate update
    vec_siz = len(alpha_init.ravel())
    step, i = np.array([np.zeros(vec_siz)]), 0  # initialize gradient-descent step to 0, iteration index in evolution

    # perform remaining iterations of gradient-descent
    alpha_evolution[0] = alpha_init
    Gt = 0  # gradient accumulator for AdaGrad normalization
    for i in range(0, max_iter - 1):
        # Print the number of iterations
        print(f"Number of iteration: {i + 1}")
        print(f"alpha {i + 1}: {alpha_evolution[i]}")

        # calculate grad and update cost function
        grad_evolution[i], cost_evolution[i] = grad_fun(alpha_evolution[i]), cost_fun(alpha_evolution[i])

        # check convergence
        if i > max(min_iter, 0) and np.abs(cost_evolution[i] - cost_evolution[i - 1]) <= tol:
            break
        else:
            # update learning rate and advance according to AdaGrad
            # "vanilla" gd step
            step = - learn_rate * grad_evolution[i]
            alpha_evolution[i + 1] = alpha_evolution[i] + step


    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # LAE plotting for visualization and debug
    if False:
        import matplotlib.pyplot as plt
        fig_cost = plt.figure(figsize=(12, 8))
        plt.plot(range(0, i), cost_evolution[0:i], '.', label="Cost")
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.legend()
        plt.show(block=False)
        plt.close(fig_cost)
    # - - - - - - - - - - - - - - - - - - - - - - - - - -

    return cost_evolution, alpha_evolution, i


def get_optimal_alpha(sigma, dataset):
    """
        Get the optimal weight vector (alpha) using gradient descent.

        Parameters:
        - sigma: numpy array, the noise covariance matrix.
        - dataset: str, the dataset to use ('mnist' or 'cifar').

        Returns:
        - optimal_alpha: numpy array, the optimal weight vector.
    """

    K = 6  # ensemble size
    M = 10 # class size
    weights_init = np.ones([K, ]) / K  # Assuming K is the ensemble size
    if dataset == "mnist":
        phis = load_phis('datasets/mnist/phis_mnist.csv')
        f_xs = load_fxs('datasets/mnist/fxs_mnist.csv')
    elif dataset == "cifar":
        phis = load_phis('datasets/cifar10/phis_cifar.csv')
        f_xs = load_fxs('datasets/cifar10/fxs_cifar.csv')

    grad_fun = lambda alpha: grad_function(alpha, phis, f_xs, sigma, M)
    cost_fun = lambda alpha: cost_function(alpha, phis, f_xs, sigma, M)

    cost_evolution, alpha_evolution, stop_iter = gradient_descent(weights_init, grad_fun, cost_fun,
                                                                  max_iter=5000, min_iter=50,
                                                                  tol=1e-5, learn_rate=0.2)
    optimal_alpha = alpha_evolution[np.argmin(cost_evolution[0:stop_iter])]

    return optimal_alpha