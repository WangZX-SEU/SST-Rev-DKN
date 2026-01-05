import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def split(x):                           # S~
    n = int(x.size()[1]/2)
    x1 = x[:, :n].contiguous()
    x2 = x[:, n:].contiguous()
    return x1, x2


def merge(x1, x2):                      # M~
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class psi(nn.Module):
    def __init__(self, ci):
        super(psi, self).__init__()
        self.ci = ci
    def inverse(self, input, in_shape):
        return input[:, :in_shape[0]]

    def forward(self, input1, input2):
        input = merge(input1, input2)
        if input.shape[1] == 2 * self.ci:
            return input[:, :self.ci], input[:, self.ci:]
        else:
            padded_input = torch.nn.functional.pad(input, (0, self.ci*2-input.shape[1]))
            return padded_input[:, :self.ci], padded_input[:, self.ci:]

class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, in_shape, stride=1, first=False, dropout_rate=0.):
        """ build invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.in_shape = in_shape

        self.psi = psi(self.in_ch)

        layers = []

        layers.append(nn.Linear(in_ch, out_ch))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(out_ch, out_ch))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(out_ch, out_ch))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(out_ch, in_ch))

        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        x1 = x[0]
        x2 = x[1]
        if self.first:
            x1, x2 = self.psi(x1, x2)
        Fx2 = self.bottleneck_block(x2)
        y1 = x1 + Fx2
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2 = x[0]
        y1 = x[1]
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.first:
            return self.psi.inverse(merge(x1, x2), self.in_shape)
        return (x1, x2)


class iRevNet(nn.Module):
    def __init__(self, c_i, h_i, dropout_rate, affineBN, in_shape):
        super(iRevNet, self).__init__()
        self.c_i = c_i
        self.h_i = h_i
        self.dropout_rate = dropout_rate
        self.in_shape = in_shape
        self.first = True

        self.stack = self.irevnet_stack(irevnet_block, c_i, h_i, dropout_rate=dropout_rate)

    def irevnet_stack(self, _block, c_i, h_i, dropout_rate):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        for c, h in zip(c_i, h_i):
            block_list.append(_block(c, h, self.in_shape, first=self.first, dropout_rate=dropout_rate))
            self.first = False
        return block_list


    def encoder(self, x):
        """ irevnet forward """
        x = x.to(torch.float64)
        length = x.shape[1]
        iter_count = 0

        out_bij = np.zeros((x.shape[1], x.shape[0], 2 * self.c_i[-1]))
        for iter_count in range(length):
            x_iter = x[:, iter_count, :]
            n = self.in_shape[0]
            out = (x_iter[:, :n-1], x_iter[:, n-1:])
            for block in self.stack:
                out = block.forward(out)
            out_bij[iter_count, :, :] = merge(out[0], out[1]).detach().numpy()
        return torch.tensor(out_bij, requires_grad=True)

    def decoder(self, out_bij):
        """ irevnet inverse """
        length = out_bij.shape[0]
        iter_count = 0
        out_bij = out_bij.reshape(1, -1)
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)

        return out


if __name__ == '__main__':
    model = iRevNet(c_i=[8, 8, 16, 16], h_i=[64, 64, 128, 128], dropout_rate=0.1, affineBN=True,
                    in_shape=[2])
    y = model(Variable(torch.randn(1, 2)))
