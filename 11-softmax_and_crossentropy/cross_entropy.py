import torch
import torch.nn as nn
import numpy as np

# def cross_entropy(actual, predicted):
#     loss = -np.sum(actual * np.log(predicted))
#     return loss # / float(predicted.shape[0])
#
# # y must be one hot encoded
# # if class 0: [1 0 0]
# # if class 1: [0 1 0]
# # if class 2: [0 0 1]
# Y = np.array([1, 0, 0])
#
# # y_pred has probabilities
# Y_pred_good = np.array([0.7, 0.2, 0.1])
# Y_pred_bad = np.array([0.1, 0.3, 0.6])
# l1 = cross_entropy(Y, Y_pred_good)
# l2 = cross_entropy(Y, Y_pred_bad)
# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2:.4f}')

# how to do the above thing in pytorch

loss = nn.CrossEntropyLoss()

# 3 samples
y = torch.tensor([2, 0, 1])
# nsamples x nclasses = 1x3
y_pred_good = torch.tensor([[2.0, 1.0, 3.1], [2.0, 1.0, 0.1], [2.0, 3.0, 0.1]])
y_pred_bad = torch.tensor([[3.5, 2.0, 0.3], [0.0, 1.0, 0.1], [2.0, 1.0, 0.1]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
print(predictions1)
print(predictions2)