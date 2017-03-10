
import torch
from torch.autograd import Variable

# task : compute d(||x||^2)/dx

x = Variable(torch.range(1,5), requires_grad=True)
print(x.data)

f = x.dot(x)
print(f.data)

f.backward()
print(x.grad)
