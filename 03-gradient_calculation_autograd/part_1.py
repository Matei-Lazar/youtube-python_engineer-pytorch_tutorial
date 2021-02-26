import torch

x = torch.rand(3, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y*y*2
print(z)
#z = z.mean()
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32) # when the input of a function has a vector, not only scalars,
# check Jacobian matrix * v = gradients
z.backward(v) # calculates dz/dx
print(x.grad)