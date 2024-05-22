from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np


predictions = []

arr1 = [2,3,4,6]
arr2 = [10,-2,7,33]
arr3 = [7,6,4,-1]
arr4 = [1,2,3,4]
arr5 = [-1,2,-3,4]
arr6 = [8,8,8,8]

predictions.append(arr1)
predictions.append(arr2)
predictions.append(arr3)
predictions.append(arr4)
predictions.append(arr5)
predictions.append(arr6)

print(predictions)

# Convert predictions to a tensor and exclude cloud prediction
device_predictions = torch.stack(predictions)  # Shape: [num_devices, B, out_channels]

# Calculate mean across device predictions
avg_predictions = torch.mean(device_predictions, dim=0)  # Mean across devices, Shape: [B, out_channels]
print("avg:")
print(avg_predictions[0])