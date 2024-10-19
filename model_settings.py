'''import argparse
import torch
import datasets

def define_settings(dataset):
    """
    Define the settings for training and evaluation, including dataset configuration and model loading.

    Parameters:
    - dataset: str, The name of the dataset being used ('mnist' or 'cifar').

    Returns:
    - model: The loaded model.
    - test_loader: The DataLoader for the test dataset.
    - num_devices: int, The number of devices used in the dataset.

    Example:
    model, test_loader, num_devices = define_settings(dataset='mnist')
    """
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--dataset-root', default='datasets/', help='Dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--dataset', default='mnist', help='Dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')


    # Define model path based on dataset
    if dataset == "mnist":
        parser.add_argument('--model_path', default='C:\\Users\\טל\\PycharmProjects\\ddnn_updated\\model_mnist.pth',
                            help='Path to the MNIST model file')
    elif dataset == "cifar":
        parser.add_argument('--model_path', default='C:\\Users\\טל\\PycharmProjects\\ddnn_updated\\model_cifar.pth',
                            help='Path to the CIFAR model file')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load the dataset
    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    train_dataset, train_loader, test_dataset, test_loader = data
    x, _ = train_loader.__iter__().__next__()
    num_devices = x.shape[1]
    in_channels = x.shape[2]

    # Load the model
    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    return model, test_loader, train_loader, num_devices
'''

import argparse
import torch
import datasets

def define_settings():
    """
    Define the settings for training and evaluation, including dataset configuration and model loading.

    Returns:
    - model: The loaded model.
    - test_loader: The DataLoader for the test dataset.
    - num_devices: int, The number of devices used in the dataset.
    """

    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--dataset-root', default='datasets/', help='Dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--dataset', default='mnist', help='Dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('--model_path', default=None, help='Path to the model file')

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Define model path based on the dataset extracted from the terminal input
    if args.dataset == "mnist" and not args.model_path:
        args.model_path = 'C:\\Users\\טל\\PycharmProjects\\ddnn_updated\\model_mnist.pth'
    elif args.dataset == "cifar" and not args.model_path:
        args.model_path = 'C:\\Users\\טל\\PycharmProjects\\ddnn_updated\\model_cifar.pth'

    args.cuda = torch.cuda.is_available()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load the dataset
    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    train_dataset, train_loader, test_dataset, test_loader = data
    x, _ = train_loader.__iter__().__next__()
    num_devices = x.shape[1]

    # Load the model
    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    return model, args.dataset, test_loader, train_loader, num_devices
