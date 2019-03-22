import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from parse_args import parser
from parse_args import args

from dataset import ImageDataset
import model as M
import net_sphere

EuclidianLoss = nn.MSELoss()
AngleLoss = net_sphere.AngleLoss()

def train(sicnn):
    cnnh = sicnn.cnnh
    cnnr = sicnn.cnnr

    ds = ImageDataset(args, args.bs)

    for i in range(args.epochs):

def trainBatch(sicnn, inputHR, inputLR, label, optimizerCNNR, optimizerCNNH):
    """
    inputHR: HR image, Variable
    inputLR: LR image, Variable
    label: face label for HR image, Variable
    """
    SR = sicnn.cnnh(inputLR)

    #CNNR
    optimizerCNNR.zero_grad()
    catHRSR = torch.cat([inputHR, SR], 0)
    catLabel = torch.cat([label, label], 0)
    SI = sicnn.cnnr(catHRSR)
    LFR = AngleLoss(SI, catLabel)
    LFR.backward()
    optimizerCNNR.step()

    #CNNH
    optimizerCNNH.zero_grad()
    LSR = EuclideanLoss(SR, inputHR)
    LSR.backward()
    optimizerCNNH.step()




def eval():
    pass

def main():
    if args.train:
        sicnn = M.SICNN(args)
        cnnh = sicnn.cnnh
        cnnr = sicnn.cnnr




if __name__ == "__main__":
    main(args)
