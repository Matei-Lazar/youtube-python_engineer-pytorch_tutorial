import torch
import numpy as np

# a = torch.ones(2,2, dtype=torch.float16)
# print(a)
#
# b = torch.tensor([2.5, 0.1])
# print(b)
#
# a = torch.rand(2,2)
# b = torch.rand(2,2)
# z = a + b # z = torch.add(a,b) # z.add_(a).add_(b)
#
# # slicing operations
# c = torch.rand(5,3)
# print(c[:, 0])
# print(c[1,1].item())
#
# # reshape / resize
# d = torch.rand(4,4)
# e = d.view(16)
# print(e)

# #convert from numpy to tensors and reverse
# a =  torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# a.add_(1)
# print(a)
# print(b) # they point to the same place in memory
#
# c = np.ones(5)
# print(c)
# d = torch.from_numpy(c)
# print(d)

# create tensor on GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    # or
    y = torch.ones(5)
    y = y.to(device)
    # you can't convert a gpu tensor back to numpy
    z = x + y
    print(z)

x = torch.ones(5, requires_grad=True) # when you need gradients
print(x)