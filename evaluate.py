import argparse
import torch
from tqdm import tqdm

import datasets
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import pandas as pd

def test_outage(model, test_loader, num_devices, outages):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    model.eval()
    num_correct = 0
    ##flag = 0
    r = []
    for data, target in tqdm(test_loader, leave=False): #target dimentions: [32], where 32 is the batch size
        for outage in outages:
            data[:, outage] = 0
        data, target = data.to(), target.to()
        data, target = Variable(data), Variable(target)
        predictions, z = model(data)
        #r.append(z)
        cloud_pred = predictions[-1] # dimentions: [32,10] represent the 32 samples, where to each one there's an array of probability(?) to be in the i-th class
        loss = F.cross_entropy(cloud_pred, target, size_average=False).item()

        pred = cloud_pred.data.max(1, keepdim=True)[1] #dimentions: [32, 1] represent the class chosen by the cloud with max probabilty
        '''if flag < 1:
            print(cloud_pred)
            print(pred)
        flag = flag + 1'''
        correct = (pred.view(-1) == target.view(-1)).long().sum().item()
        num_correct += correct

    ##########################################################################################################333
    '''g = np.concatenate(r, axis=0) #(9984, 96, 14, 9)
    c = g.reshape(-1, g.shape[-1])

    # Convert the matrix to a Pandas DataFrame
    df = pd.DataFrame(c)

    # Specify the CSV file path
    csv_file_path = 'output_matrix.csv'

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False, header=False)

    cov_matrix = c @ c.T
    print(cov_matrix.shape)'''
    N = len(test_loader.dataset) # N = 10,000, number of iterations in the above loop: 312 (which is N/batch_size)

    return 100. * (num_correct / N)

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

    outages = []
    acc = test_outage(model, test_loader, num_devices, outages)
    print('Missing Device(s) {}: {:.4f}'.format(outages, acc))