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
    for data, target in tqdm(test_loader, leave=False): #target dimentions: [32], where 32 is the batch size
        for outage in outages:
            data[:, outage] = 0
        data, target = data.to(), target.to()
        data, target = Variable(data), Variable(target)
        predictions, z = model(data, std)
        r.append(z)
        cloud_pred = predictions[-1] # dimentions: [32,10] represent the 32 samples, where to each one there's an array of probability(?) to be in the i-th class
        loss = F.cross_entropy(cloud_pred, target, size_average=False).item()

        pred = cloud_pred.data.max(1, keepdim=True)[1] #dimentions: [32, 1] represent the class chosen by the cloud with max probabilty
        correct = (pred.view(-1) == target.view(-1)).long().sum().item()
        num_correct += correct

    ##########################################################################################################333

    X = np.concatenate(r, axis=0) #(9984, 96, 14, 9)

    '''dim1 = int(X.shape[1] / 6)  # 16
    dim2 = X.shape[2] # 14
    dim3 = X.shape[3] # 9
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                cov_of_feature(i,j,k,X)'''

    #hisogram()

    N = len(test_loader.dataset) # N = 10,000, number of iterations in the above loop: 312 (which is N/batch_size)

    return 100. * (num_correct / N)

#########################################################################################################

def cov_of_feature(i, j, k, X): #to calculate the X_ijk of a specific feature (i,j,k) and then X_ijk.T @ X_ijk = Cov_ijk
    N = X.shape[0] #Number of samples
    dim = int(X.shape[1] / 6) #16 - which is the length to "jump" between devices in the matrix X, united from h1, ... , h6
    matrix = np.zeros((N, 6))
    for r in range(N):
        for t in range(6):
            matrix[r][t] = X[r][i + dim * t][j][k]
    cov = (matrix.T @ matrix) / N
    trc = np.trace(cov)
    #return trc

    data = [[i, j, k, trc]]
    # Convert the matrix to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Specify the CSV file path
    csv_file_path = 'trc.csv'

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, mode='a', index=False, header=False)

############################################################################################################

def hisogram():
    # Read the data from the CSV file
    df = pd.read_csv('trc.csv', header=None)

    # Extract the data from the fourth column
    traces = df.iloc[:, 3]  # Assuming the index of the fourth column is 3

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
    with open("avg.txt", "w") as f:
        f.write(f"Average of the traces: {average}\n")

########################################################################################################

def create_snr_graph():
    values = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

    for index, snr in enumerate(values):
        std = np.sqrt(36.946195351075666 / (6*snr)) # std
        acc = test_outage(model, test_loader, num_devices, outages, std)
        print('SNR = {:.4f}, ACC = {:.4f}'.format(snr, acc))
        data = [[snr, acc]]
        # Convert the matrix to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Specify the CSV file path
        csv_file_path = 'snr_acc.csv'

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, mode='a', index=False, header=False)

    # Make it a graph:

    # Load the data from CSV file
    data = pd.read_csv("snr_acc.csv", header=None, names=['snr', 'acc'])

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

def present_all_snr_graphs():
    # Load the data from three CSV files
    data1 = pd.read_csv("MAX_snr_acc_local_.csv", header=None, names=['snr', 'acc'])
    data2 = pd.read_csv("AVG_snr_acc_local.csv", header=None, names=['snr', 'acc'])
    data3 = pd.read_csv("snr_global2.csv", header=None, names=['snr', 'acc'])

    # Extract SNR and ACC values
    snr1, acc1 = data1['snr'], data1['acc']
    snr2, acc2 = data2['snr'], data2['acc']
    snr3, acc3 = data3['snr'], data3['acc']

    # Plot the data on the same axis
    plt.figure(figsize=(10, 6))
    plt.plot(snr1, acc1, marker='o', linestyle='-', color='b', label='Local - Max')
    plt.plot(snr2, acc2, marker='s', linestyle='-', color='g', label='Local - Avg')
    plt.plot(snr3, acc3, marker='^', linestyle='-', color='r', label='Global')

    # Add title and labels
    plt.title('SNR vs Accuracy')
    plt.xlabel('SNR')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Add legend
    plt.legend()

    # Set the number of ticks
    num_ticks_x = 9
    num_ticks_y = 11

    # Generate evenly spaced ticks for x-axis and y-axis
    x_ticks = np.linspace(0, 0.2, num_ticks_x)
    y_ticks = np.linspace(0, 100, num_ticks_y)

    # Set the ticks on x-axis and y-axis
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # Show the plot
    plt.show()



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
    '''
    for i in range(num_devices):
        outages = [i]
        acc = test_outage(model, test_loader, num_devices, outages)
        print('Missing Device(s) {}: {:.4f}'.format(outages, acc))

    for i in range(1, num_devices + 1):
        outages = list(range(i, num_devices))
        acc = test_outage(model, test_loader, num_devices, outages)
        print('Missing Device(s) {}: {:.4f}'.format(outages, acc))
    '''

    outages = [] # All devices included
    #create_snr_graph()
    present_all_snr_graphs()
