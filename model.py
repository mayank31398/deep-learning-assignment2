import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Variable


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 16, 240, 320
        self.l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 4)),
            nn.Conv2d(16, 16, kernel_size=5),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )
        
        # 16, 38, 38
        self.l2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=5),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2)
        )

        # 8, 17, 17
        self.l3 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=5, stride=(2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(4),
        )

        # 4, 7, 7
        self.l4 = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=5),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(2),
        )

        # 2, 3, 3
        self.l5 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3),
            nn.Sigmoid()
        )
        # 1, 1, 1

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        return x.view(-1)


class ConvLSTM_cell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.Wxi = nn.Conv2d(input_channels, hidden_channels,
                             kernel_size, 1, padding, bias=True)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels,
                             kernel_size, 1, padding, bias=False)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels,
                             kernel_size, 1, padding, bias=True)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels,
                             kernel_size, 1, padding, bias=False)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels,
                             kernel_size, 1, padding, bias=True)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels,
                             kernel_size, 1, padding, bias=False)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels,
                             kernel_size, 1, padding,  bias=True)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels,
                             kernel_size, 1, padding, bias=False)

    def forward(self, x, h, c):
        ci = th.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = th.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * th.tanh(self.Wxc(x) + self.Whc(h))
        co = th.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * th.tanh(cc)

        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        self.Wci = Variable(th.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wcf = Variable(th.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wco = Variable(th.zeros(1, hidden, shape[0], shape[1])).cuda()

        h = th.zeros(batch_size, hidden, shape[0], shape[1]).cuda()
        c = th.zeros(batch_size, hidden, shape[0], shape[1]).cuda()

        return h, c


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size
    ):
        super().__init__()

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTM_cell(
                self.input_channels[i], self.hidden_channels[i], kernel_size).cuda()
            setattr(self, name, cell)
        
        # CNN
        self.cnn = ConvNet().cuda()

    def forward(self, inp):
        internal_state = []
        outputs = []
        steps = inp.shape[1]

        for step in range(steps):
            x = inp[:, step, ...]
            x = th.from_numpy(x).cuda()

            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if(step == 0):
                    batch_size, _, height, width = x.shape
                    h, c = getattr(self, name).init_hidden(batch_size=batch_size,
                                                           hidden=self.hidden_channels[i],
                                                           shape=(height, width))
                    internal_state.append((h, c))

                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            
            x = self.cnn(x)
            outputs.append(x.view(x.shape[0], 1))
        
        outputs = th.cat(outputs, dim=1)

        return outputs
