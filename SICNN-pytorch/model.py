import torch
import torch.nn as nn
import torch.nn.functional as F

class SICNN(nn.Module):
    def __init__(self, args):
        super(SICNN, self).__init__()
        self.cnnh = CNNH(args)
        self.cnnr = CNNR(args)

    def forward(self, input):
        """
        Args:
            input: LR image

        Returns:
            feature vector
        """

        SR = cnnh(input)
        SI = cnnr(SR)
        return SI

class CNNH(nn.Module):
    def __init__(self.args):
        super(CNNH, self).__init__()

    def forward(self, input):
        pass

class CNNR(nn.Module):
    def __init__(self.args):
        super(CNNR, self).__init__()

    def forward(self, input):
        pass
