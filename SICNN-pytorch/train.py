import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from parse_args import parser
from parse_args import args

import model as M

def train(args):
    pass

def eval(args):
    pass

def main(args):
    if args.train:
        sicnn = M.SICNN(args)
        cnnh = sicnn.cnnh
        cnnr = sicnn.cnnr
        


if __name__ == "__main__":
    main(args)
