#!/usr/bin/env python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.nn.init as init

import timeit
import matplotlib.pyplot as plt
import numpy as np
# uncomment in jupyter notebook
# %matplotlib inline


time1 = timeit.default_timer()

# Training settings
# for terminal use. In notebook, you can't parse arguments
parser = argparse.ArgumentParser(description='ELEG5491 A2 Image Classification on CIFAR-10')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--conv1-width', type=int, default=6, metavar='N',
                    help='the output channel of the first Conv layer')
# parser.add_argument('--pool-size', type=int, default=2, metavar='N',
#                     help='the kernel size of pooling size, the size is pool-size*pool-size')
parser.add_argument('--dropout-fc', action='store_true', default=False,
                    help='adding dropout after each fully connected layer')

parser.add_argument('--batch-Normalize', action='store_true', default=False,
                    help='adding batch normalization layer after each layer, '
                         'except the very last layer for classification')

parser.add_argument('--kaiming-weight-init', action='store_true', default=False,
                    help='using kaiming_normal to initial the weight of each layer')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)
# setting argument for terminal
in_train_batch_size = args.batch_size
in_test_batch_size = args.test_batch_size
in_epochs = args.epochs
in_lr = args.lr
in_momentum = args.momentum
in_seed = args.seed
in_log_interval = args.log_interval
in_dropout = args.dropout_fc
in_batchNormalize = args.batch_Normalize
in_kaiming_weight_init = args.kaiming_weight_init

#the assinment ask set this to 6, change this see the super-efficiency of GPU acceleration
in_conv1_width = args.conv1_width
in_pool_size = 2 #args.pool_size need change codes to auto get the size of pooling result

in_enable_cuda = args.cuda

# assign the random seed, it's better to set random seed based on time
torch.manual_seed(in_seed)
if in_enable_cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if in_enable_cuda else {}
print('parameters for cuda: ',kwargs)

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=in_train_batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=in_test_batch_size,
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

###################################################
# Let us show some of the training images, for fun.
# functions to show an image
# import matplotlib.pyplot as plt
# import numpy as np
# # %matplotlib inline
# def imshow(img):
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#
# # show some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s'%classes[labels[j]] for j in range(4)))


def draw_training_curve_and_accuarcy():
    plt.figure(2)
    epoch_index = list(range(1,len(training_error)+1))
    color_linestyle = 'b-'
    label = 'training error'
    plt.plot(epoch_index,training_error,color_linestyle , label=label, linewidth=1.5)

    color_linestyle = 'r-.'
    label = 'test accuracy'
    plt.plot(epoch_index,test_accuracy,color_linestyle , label=label, linewidth=1.5)

    plt.xlabel("epoches ")
    plt.ylabel("training error and test accuracy")
    plt.title(" ELEG5491 A2 Image Classification on CIFAR-10 elevation")
    plt.ylim(0, 2.5)
    plt.legend()
    # plt.show()
    plt.savefig('training_curve_and_accuarcy.jpg')
    pass


# codes from tutorial 3:  Deep learning toolkits I
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    plt.figure(1)
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    plt.savefig('weight.jpg')
    # plt.show()



###################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: define your network here
        self.conv1 = nn.Conv2d(3, in_conv1_width, kernel_size=5,stride=1)
        if in_kaiming_weight_init:
            # https://github.com/alykhantejani/nninit
            init.kaiming_normal(self.conv1.weight) # using fan_in, only consider the forward init
            # init.constant(self.conv1.bias, 0.1)
        if in_batchNormalize:
            self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=in_pool_size, stride=2)
        self.pool1_relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_conv1_width, 16, kernel_size=5, stride=1)
        if in_kaiming_weight_init:
            init.kaiming_normal(self.conv2.weight)
        if in_batchNormalize:
            self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=in_pool_size, stride=2)
        self.pool2_relu=nn.ReLU()

        # the size of pool2 result is torch.Size([4, 16, 3, 3]), so is should be 3*3*16, when pooling kernel size is 4
        # the size of pool2 result is torch.Size([4, 16, 5, 5]), so is should be 5*5*16, when pooling kernel size is 2
        self.fc1 = nn.Linear(5*5*16, 120)
        if in_kaiming_weight_init:
            init.kaiming_normal(self.fc1.weight)
        if in_batchNormalize:
            self.bn_fc1 = nn.BatchNorm1d(120)
        if in_dropout:
            self.dropout1 = nn.Dropout()
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        if in_kaiming_weight_init:
            init.kaiming_normal(self.fc2.weight)
        if in_batchNormalize:
            self.bn_fc2 = nn.BatchNorm1d(84)
        if in_dropout:
            self.dropout2 = nn.Dropout()
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(84, 10)
        self.relu_fc3 = nn.ReLU()

    def forward(self, x):
        # TODO

        x = self.conv1(x)  # When a nn.Module is called, it will compute the result
        # print('size of result in conv1: ',x.size())
        if in_batchNormalize:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.pool1_relu(x)

        # print('result of pool1: ', x)
        x = self.conv2(x)
        if in_batchNormalize:
            x = self.bn2(x)
        x = self.relu2(x)
        # print('result of conv2: ', x)
        x = self.pool2(x)
        x = self.pool2_relu(x)
        # print('result of pool2: ', x)

        # x_size = x.size()
        # print(x_size)
        # print(x.size(0))
        x = x.view(x.size(0), -1)
        # x_size = x.size()
        # print(x_size)

        x = self.fc1(x)
        if in_batchNormalize:
            x = self.bn_fc1(x)
        # print('result of fc1: ', x)
        if in_dropout:
            x = self.dropout1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        if in_batchNormalize:
            x = self.bn_fc2(x)
        # print('result of fc2: ', x)
        if in_dropout:
            x = self.dropout2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        # print('result of fc: ', x)
        x = self.relu_fc3(x)
        return x


model = Net()
print(model)
if in_enable_cuda:
    model.cuda()
print(model)

optimizer = optim.SGD(model.parameters(), lr=in_lr, momentum=in_momentum,weight_decay=0.0005)

training_error = []
test_accuracy = []

def train(epoch):
    model.train()
    average_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if in_enable_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)   # is it true to use such a loss over cross-entropy loss?
        # when use nll_loss as lost function, the loss value become NaN in few step, so it didn't work
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # print(loss.data)
        average_loss += loss.data[0]*in_train_batch_size
        if batch_idx % in_log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    print('\n Total count: {}, Average loss: {:.6f}'.format(
        len(train_loader.dataset),average_loss/len(train_loader.dataset)))

    training_error.append(average_loss/len(train_loader.dataset))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if in_enable_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # test_loss += F.nll_loss(output, target).data[0]
        test_loss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy.append(1.0*correct / len(test_loader.dataset))


for epoch in range(1, in_epochs + 1):
    train(epoch)
    test(epoch)

# filters = model.conv1.weight.data.view(3*6,5,5)
filters = model.conv1.weight.data.cpu().view(3*6,5,5)
vis_square(filters.numpy())

# vis_square(filters.numpy())
# if in_enable_cuda is False:
#     vis_square(filters.numpy())
# else:
#     print('numpy conversion for FloatTensor is not supported in CUDA, so no filter learned showed')

# print training curve and test accuracy for at least 5 epoches
draw_training_curve_and_accuarcy()

# plt.show()



time2 = timeit.default_timer()
print('cost time of whole process: %.2f minutes\n' % ((time2 - time1) / 60.0))