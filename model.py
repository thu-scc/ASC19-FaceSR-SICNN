import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
        -> Conv -> PReLU ->
    """
    def __init__(self, ins, outs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(ins, outs, (3,3), (1,1), padding=1) # different weight
        self.relu1 = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        return out


class ResBlock(nn.Module):
    """
        -> -> Conv -> PReLU -> Conv -> PReLU -> ADD ->
           |                                     |
           +-------------------------------------+
    """
    def __init__(self, ins, outs):
        super(ResBlock, self).__init__()
        self.basic1 = BasicBlock(ins,ins)
        self.basic2 = BasicBlock(ins, ins)

    def forward(self, x):
        residual = x
        out = self.basic1(x)
        out = self.basic2(out)
        out += residual
        return out

class DenseBlock(nn.Module):
    """
    -> Conv -> PReLU -> [-> Conv -> PReLU -> CAT ->]*6 ->
                         |                    |
                         +--------------------+
    """
    def __init__(self, ins):
        super(DenseBlock, self).__init__()
        self.basic1 = BasicBlock(ins, 64)
        self.basic2 = BasicBlock(64, 32)
        self.basic3 = BasicBlock(96, 32)
        self.basic4 = BasicBlock(128, 32)
        self.basic5 = BasicBlock(160, 32)
        self.basic6 = BasicBlock(192, 32)
        self.basic7 = BasicBlock(224, 32)

    def forward(self, x):
        x = self.basic1(x)
        y1 = self.basic2(x)
        y1 = torch.cat((x, y1), 1)
        x = self.basic3(y1)
        y1 = torch.cat((x, y1), 1)
        x = self.basic4(y1)
        y1 = torch.cat((x, y1), 1)
        x = self.basic5(y1)
        y1 = torch.cat((x, y1), 1)
        x = self.basic6(y1)
        y1 = torch.cat((x, y1), 1)
        x = self.basic7(y1)
        y1 = torch.cat((x, y1), 1)
        return y1

class CNNHNet(nn.Module):
    """
    Hallucination network. convert LR images to an SR images
    """
    def __init__(self, upscale_factor, batch_size):
        super(CNNHNet, self).__init__()
        self.batchsize = batch_size
        self.dense1 = DenseBlock(3)
        self.deconv1 = nn.ConvTranspose2d(256, 256, (3,3), (2,2), output_padding=1, padding=1)
        self.relude1 = nn.PReLU()

        self.dense2 = DenseBlock(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, (5,5), (2,2), output_padding=1, padding=2)
        self.relude2 = nn.PReLU()

        #self.dense3 = DenseBlock(256)
        #self.deconv3 = nn.ConvTranspose2d(256, 256, (3,3), (1,1), output_padding=0, padding=1)
        #self.relude3 = nn.PReLU()

        self.prebasic4_1 = BasicBlock(256, 64)
        self.prebasic4_2 = BasicBlock(64, 32)
        self.prebasic4_3 = BasicBlock(96, 32)
        self.gen = nn.Conv2d(128, 3, (5,5), (1,1), padding=2)
        self.tanh = nn.Tanh()

    def forward(self, input):
        """
         Args:
            input: LR images
         Returns:
            output: SR images
        """
        y1 = self.dense1(input)
        x = self.relude1(self.deconv1(y1))
        y1 = self.dense2(x)
        x = self.relude2(self.deconv2(y1))
        #y1 = self.dense3(x)
        #x = self.relude3(self.deconv3(y1))

        x = self.prebasic4_1(x)
        y1 = self.prebasic4_2(x)
        y1 = torch.cat((x, y1), 1)
        x = self.prebasic4_3(y1)
        x = torch.cat((x, y1), 1)

        output = self.tanh(self.gen(x))
        return output

    def _initialize_weights(self):
        return null
