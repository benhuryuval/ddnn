
from __future__ import print_function

import torch


import torch.nn as nn
import torch.nn.functional as F

import pandas as pd




def _layer(in_channels, out_channels, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResLayer, self).__init__()
        self.c1 = _layer(in_channels, out_channels)
        self.c2 = _layer(out_channels, out_channels, activation=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        
        # residual connection
        if x.shape[1] == h.shape[1]:
            h += x

        h = self.activation(h)

        return h


class DeviceModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeviceModel, self).__init__()
        self.model = nn.Sequential( #layers of each device
            ResLayer(in_channels, 8),
            ResLayer(8, 8),
            ResLayer(8, 16),
            ResLayer(16, 16),
            ResLayer(16, 16),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, out_channels)
    
    def forward(self, x):
        B = x.shape[0]
        h = self.model(x)
        p = self.pool(h)
        return h, self.classifier(p.view(B, -1))
    

class DDNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_devices):
        super(DDNN, self).__init__()
        self.num_devices = num_devices
        self.device_models = []
        for _ in range(num_devices):
            self.device_models.append(DeviceModel(in_channels, out_channels))
        self.device_models = nn.ModuleList(self.device_models)

        cloud_input_channels = 16*num_devices
        self.cloud_model = nn.Sequential( #layers of the cloud
            ResLayer(cloud_input_channels, 64),
            ResLayer(64, 64),
            ResLayer(64, 128),
            nn.AvgPool2d(2, 2),
            ResLayer(128, 128),
            ResLayer(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, out_channels) #out_channels = 10

    def forward(self, x, sigma, exit, alpha, dataset, export_info="no"):
        """
            Forward pass for device and cloud models with noise injection based on the device-specific variance.

            Args:
                x (torch.Tensor): Input data with shape [B, num_devices, channels, height, width].
                sigma (torch.Tensor): Covariance matrix for noise injection, shape [num_devices, num_devices].
                exit (str): Specifies whether to use "global" or "local" exit.
                alpha (list or torch.Tensor): Weights for weighted aggregation of device predictions.

            Returns:
                list: Predictions for each device (and cloud if "global" exit is used).

            Example:
                preds = model.forward(x, sigma, exit="local", alpha=[0.5, 0.5])
        """

        # Initialize batch size and lists to store device outputs and predictions
        B = x.shape[0]  # Batch size
        hs, predictions = [], []

        # Iterate over each device model to get the output and prediction
        for i, device_model in enumerate(self.device_models):
            h, prediction = device_model(x[:, i])  # h: device output, prediction: device prediction

            # Add noise to the device output or prediction based on the exit type (global or local)
            device_noise_std = torch.sqrt(torch.tensor(sigma[i, i], dtype=torch.float32))
            if exit == "global":  # Global exit: add noise to the device output
                noise = torch.normal(0, device_noise_std, h.size())
                h = h + noise
                hs.append(h)
                predictions.append(prediction)

            elif exit == "local":  # Local exit: add noise to the device prediction
                noise = torch.normal(0, device_noise_std, prediction.size())
                prediction = prediction + noise
                hs.append(h)
                predictions.append(prediction)

            # Write predictions to the CSV file for each sample and class
            if export_info == "yes":
                for sample_idx in range(B):
                    for class_idx in range(prediction.size(1)):  # Assuming prediction has shape [B, num_classes]
                        # Create a DataFrame for the current sample's prediction
                        data = {
                            'Sample': sample_idx,
                            'Device': i,
                            'Class': class_idx,
                            'Prediction': prediction[sample_idx, class_idx].item()
                        }
                        df = pd.DataFrame([data])

                        # Write the DataFrame to CSV in append mode
                        df.to_csv(f"phis_{dataset}.csv", mode='a', header=False, index=False)

        # For global exit: pass the concatenated device outputs through the cloud model and classify
        if exit == "global":
            h = torch.cat(hs, dim=1)  # Concatenate device outputs
            h = self.cloud_model(h)  # Cloud model processing
            h = self.pool(h)  # Apply pooling
            prediction = self.classifier(h.view(B, -1))  # Classification step
            predictions.append(prediction)

        # For local exit: perform weighted aggregation of device predictions
        elif exit == "local":
            probabilities = weighted_aggregation(predictions, alpha, dataset, export_info)  # Weighted aggregation
            predictions.append(probabilities)

        # Return predictions for both local and global exit cases
        return predictions




def weighted_aggregation(predictions, alpha, dataset, export_info="no"):
    """
    Perform weighted aggregation of predictions from multiple devices using alpha weights.

    Args:
        predictions (list of torch.Tensor): Predictions from each device. Shape: [num_devices, B, out_channels]
        alpha (list or torch.Tensor): Weights for each device. Shape: [num_devices]
        dataset (str): The name of the dataset for creating the CSV filename.
        export_info (str): If "yes", write the probabilities to a CSV file.

    Returns:
        torch.Tensor: Aggregated class probabilities. Shape: [B, out_channels]

    Example:
        probs = weighted_aggregation([pred1, pred2], [0.6, 0.4], "train_dataset", export_fxs="yes")
    """
    # Convert predictions to a tensor and exclude cloud prediction
    device_predictions = torch.stack(predictions)  # Shape: [num_devices, B, out_channels]

    # Reshape alpha to match the shape for broadcasting
    alpha_tensor = torch.tensor(alpha, dtype=device_predictions.dtype, device=device_predictions.device).view(-1, 1, 1)  # Shape: [num_devices, 1, 1]

    # Calculate weighted average across device predictions
    weighted_sum = torch.sum(device_predictions * alpha_tensor, dim=0)  # Weighted sum across devices, Shape: [B, out_channels]
    normalization_factor = torch.sum(alpha_tensor, dim=0)  # Normalization factor to ensure weighted avg, Shape: [1, 1]

    # Divide by the normalization factor to get the weighted average
    weighted_avg_predictions = weighted_sum / normalization_factor  # Shape: [B, out_channels]

    # Apply softmax to weighted_avg_predictions along the last dimension
    probabilities = F.softmax(weighted_avg_predictions, dim=1)  # Shape: [B, out_channels]

    # Export probabilities to CSV if export_fxs is enabled
    if export_info == "yes":
        # Create the filename for the probabilities
        prob_filename = f"fxs_{dataset}_prob.csv"

        # Convert probabilities to a DataFrame
        df_probabilities = pd.DataFrame(probabilities.cpu().detach().numpy())

        # Write the probabilities to the CSV file
        df_probabilities.to_csv(prob_filename, mode='a', header=False, index=False)

    return probabilities


def MAX_aggregation(predictions):
    """
        Perform MAX aggregation of predictions from multiple devices.

        Args:
            predictions (list of torch.Tensor): Predictions from each device. Shape: [num_devices, B, out_channels]

        Returns:
            torch.Tensor: Aggregated class probabilities based on the maximum prediction. Shape: [B, out_channels]

        Example:
            probs = MAX_aggregation([pred1, pred2])
    """
    # Convert predictions to a tensor and exclude cloud prediction
    device_predictions = torch.stack(predictions)  # Shape: [num_devices, B, out_channels]

    # Calculate MAX across device predictions
    max_predictions, _ = torch.max(device_predictions, dim=0)  # MAX across devices, Shape: [B, out_channels]

    # Apply softmax to MAX_predictions along the last dimension
    probabilities = F.softmax(max_predictions, dim=1)  # Shape: [B, out_channels]

    return probabilities

