#encoding=utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import os
if os.path.exists('./step3/cnn.pkl'):
    os.remove('./step3/cnn.pkl')
#加载数据
train_data = torchvision.datasets.MNIST(
    root='./step3/mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(
    ),  # Converts a PIL.Image or numpy.ndarray to                                                    
    download=False,
)
#取6000个样本为训练集
train_data_tiny = []
for i in range(6000):
    train_data_tiny.append(train_data[i])
train_data = train_data_tiny
#********* Begin *********#

# mini_batch ***
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)


#构建卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=
                2,  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(
                kernel_size=2
            ),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7,
                             10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(
            x.size(0),
            -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
#构建优化器与损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
#训练模型
for step, (x, y) in enumerate(train_loader):
    b_x = Variable(x)
    b_y = Variable(y)
    output = cnn(b_x)
    loss = loss_func(output, b_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#********* End *********#
#保存模型
torch.save(cnn.state_dict(), './step3/cnn.pkl')