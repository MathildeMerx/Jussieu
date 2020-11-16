# pip install -r http://webia.lip6.fr/~bpiwowar/requirements-amal.txt
import torch
from tp1 import mse, LinearFunction
from torch.autograd import gradcheck
# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)


test1 = torch.autograd.gradcheck(mse, (yhat, y), eps=1e-6, atol=1e-4)
print(test1)
#  TODO:  Test du gradient de Linear




# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
linear = LinearFunction.apply

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
test2 = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test2)















