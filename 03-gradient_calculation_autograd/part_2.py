import torch

x = torch.rand(3, requires_grad=True)
print(x)

#how to prevent pytorch from tracking the history and calculate the gradients

# x.requires_grad_(False)
# x.detach()
with torch.no_grad():
    y = x + 2
    print(y)