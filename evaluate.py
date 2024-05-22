import argparse
import torch
from tqdm import tqdm

import datasets
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_outage(model, test_loader, num_devices, outages, std):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    model.eval()
    num_correct = 0
    r = []
    for data, target in tqdm(test_loader, leave=False): #target dimentions(represents the ground-truth label): [32], where 32 is the batch size
        for outage in outages:
            data[:, outage] = 0
        data, target = data.to(), target.to()
        data, target = Variable(data), Variable(target)

        '''if flag == 0:
            predictions = model(data, std, 0)
        if flag == 1:
            predictions = model(data, std, 1)
            print("targets:")
            print(target)
            flag = 0'''

        predictions, z = model(data, std)
        r.append(z)

        local_pred = predictions[-1] # dimentions: [32,10] represent the 32 samples, where to each one there's an array of probability(?) to be in the i-th class
        loss = F.cross_entropy(local_pred, target, size_average=False).item()

        local_pred = local_pred.data.max(1, keepdim=True)[1] #dimentions: [32, 1] represent the class chosen by the cloud with max probabilty
        correct = (local_pred.view(-1) == target.view(-1)).long().sum().item()
        num_correct += correct

    ##########################################################################################################333
    '''
    X = np.concatenate(r, axis=0) #[N~9984, 6*10]

    dim1 = int(X.shape[1] / 6)  #
    for i in range(dim1):
        cov_of_feature(i, X)

    hisogram()
    '''

    N = len(test_loader.dataset) # N = 10,000, number of iterations in the above loop: 312 (which is N/batch_size)

    return 100. * (num_correct / N)

#########################################################################################################

def cov_of_feature(i, X): #to calculate the X_i of a specific feature (i) and then X_i.T @ X_i = Cov_i
    N = X.shape[0] #Number of samples
    dim = int(X.shape[1] / 6) #16 - which is the length to "jump" between devices in the matrix X, united from h1, ... , h6
    matrix = np.zeros((N, 6))
    for r in range(N):
        for t in range(6):
            matrix[r][t] = X[r][i + dim * t]
    cov = (matrix.T @ matrix) / N
    trc = np.trace(cov)
    #return trc

    data = [[i, trc]]
    # Convert the matrix to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Specify the CSV file path
    csv_file_path = 'trc_local.csv'

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, mode='a', index=False, header=False)

############################################################################################################

def hisogram():
    # Read the data from the CSV file
    df = pd.read_csv('trc_local.csv', header=None)

    # Extract the data from the fourth column
    traces = df.iloc[:, 1]  # Assuming the index of the fourth column is 1

    # Plot the histogram
    plt.hist(traces, bins=100)  # Adjust the number of bins as needed
    plt.xlabel('Trace Value')
    plt.ylabel('Number of features')
    plt.title('Histogram of the number of features against their covariance matrix trace value')
    plt.show()

    # Remove NaN values if any
    traces = traces.dropna()

    # Calculate the average using pandas
    average = traces.mean()

    # Sanity check
    num_rows = len(traces)
    print("Number of rows extracted:", num_rows)

    print("Average of the trace of covariance features matrixs:", average) # mnist average : 36.946195351075666

    # Write average to a text file
    with open("avg_local.txt", "w") as f:
        f.write(f"Average of the traces: {average}\n")

########################################################################################################

def create_snr_graph():
    values = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

    for index, snr in enumerate(values):
        std = np.sqrt(112.82436884928032 / (6*snr)) # std
        acc = test_outage(model, test_loader, num_devices, outages, std)
        print('SNR = {:.4f}, ACC = {:.4f}'.format(snr, acc))
        data = [[snr, acc]]
        # Convert the matrix to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Specify the CSV file path
        csv_file_path = 'snr_acc_local.csv'

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, mode='a', index=False, header=False)

    # Make it a graph:

    # Load the data from CSV file
    data = pd.read_csv("snr_acc_local.csv", header=None, names=['snr', 'acc'])

    # Extract SNR and ACC values
    snr = data['snr']
    acc = data['acc']

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(snr, acc, marker='o', linestyle='-')
    plt.title('SNR vs Accuracy')
    plt.xlabel('SNR')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Set the number of ticks
    num_ticks_x = 9
    num_ticks_y = 11

    # Generate evenly spaced ticks for x-axis and y-axis
    x_ticks = np.linspace(0, 0.2, num_ticks_y)
    y_ticks = np.linspace(0, 100, num_ticks_y)

    # Set the ticks on x-axis and y-axis
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.show()

########################################################################################################

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='C:\\Users\\Yuval\\Documents\\GitHub\\ddnn\\models\\mnist.pth',
                        help='output directory')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    train_dataset, train_loader, test_dataset, test_loader = data
    x, _ = train_loader.__iter__().__next__()
    num_devices = x.shape[1]
    in_channels = x.shape[2]
    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    outages = [] # All devices included
    create_snr_graph()

