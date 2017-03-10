
from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms

import numpy as np
import torch
import timeit

d =3000

#using numpy
A = np.random.rand(d,d).astype(np.float32)
B = np.random.rand(d,d).astype(np.float32)
time1 = timeit.default_timer()
C = A.dot(B)
time2 = timeit.default_timer()
print('cost time of numpy: %f\n'%(time2-time1))
#print(C)

A = torch.rand(d,d).cuda()
B = torch.rand(d,d).cuda()
time2 = timeit.default_timer()
C = torch.mm(A,B)
time3 = timeit.default_timer()
print('cost time of torch: %f\n'%(time3-time2))
#print(C)
