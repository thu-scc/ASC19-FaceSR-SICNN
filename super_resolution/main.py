from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net,SICNNNet
from data import get_training_set, get_test_set

# ASC loss
import net_sphere


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int,default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--alpha', type=float, default=0.1, help='alpha to combine LSR and LSI in the paper Algorithm 1')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False, drop_last=True)

print('===> Building model')
model = SICNNNet(upscale_factor=opt.upscale_factor, batchsize=opt.batchSize).to(device)
# criterion = nn.MSELoss()
judgenet = getattr(net_sphere, 'sphere20a')()
judgenet.load_state_dict(torch.load('sphere20a.pth'))
judgenet.feature = True

for param in judgenet.parameters():
    param.requires_grad = False
# judgenet.train()
pjnet = judgenet.cuda()
# pjnet = nn.DataParallel(judgenet).cuda()

def pjnet_loss_fn(output, target, batchsize):
    newimg = torch.cat((output, target), 0)
    res = pjnet(newimg)
    cosdistance = res[0].dot(res[1])/(res[0].norm()*res[1].norm())
    for i in range(1, batchsize):
        cosdistance += res[i].dot(res[i+batchsize])/(res[i].norm()*res[i + batchsize].norm())
    return 1 - cosdistance/batchsize
# criterion = pjnet_loss_fn



optimizer_CNNR = optim.Adam(model.cnnr.parameters(), lr=opt.lr)
optimizer_CNNH = optim.Adam(model.cnnh.parameters(), lr=opt.lr)
EuclideanLoss = nn.MSELoss()
AngleLoss = net_sphere.AngleLoss()


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 100):
        input, target = batch[0].to(device), batch[1].to(device)
        newlabel = torch.cat((target, target), 0) #new label

        optimizer_CNNR.zero_grad()
        optimizer_CNNH.zero_grad()
        # print(input.shape)
        SR_data, SI_embed_HR, SI_embed_SR, SI_angular = model(input, target)

        #CNNR
        LFR = AngleLoss(SI_angular, newlabel)
        LFR.backward()
        optimizer_CNNR.step()

        #CNNH
        LSR = EuclideanLoss(SR_data, input)
        LSI = EuclideanLoss(SI_embed_HR, SI_embed_SR)
        L = LSR + args.alpha * LSI
        L.backward()
        optimizer_CNNH.step()

        # loss = pjnet_loss_fn(output, target, opt.batchSize)
        # epoch_loss += output.item()
        # output.backward()
        # optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), output.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            loss_num = model(input, target)
            avg_loss = 0.0
            # loss_num = pjnet_loss_fn(prediction, target, opt.testBatchSize)
            # mse = criterion(prediction, target)
            # psnr = 10 * log10(1 / mse.item())
            avg_loss += loss_num
    print("===> Avg. IS: {:.4f} ".format(1 - avg_loss / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)
