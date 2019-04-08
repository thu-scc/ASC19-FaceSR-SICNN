from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net, SICNNNet
from data import get_training_set, get_test_set

# ASC loss
import net_sphere


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int,default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=64, help='testing batch size')# todo
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--judgeloss', type=float, default = 0.0, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')


parser.add_argument('--alpha', type=float, default=10.0, help='alpha to combine LSR and LSI in the paper Algorithm 1')
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
model = SICNNNet(upscale_factor=opt.upscale_factor, batch_size=opt.batchSize).to(device)
# print(model.cpu())
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
        if iteration > 200:
            break
        input, target = batch[0].to(device), batch[1].to(device)
        

        optimizer_CNNR.zero_grad()
        optimizer_CNNH.zero_grad()

        # SR_data, SI_embed_HR, SI_embed_SR, SI_angular, fea1, fea2 = model(input, target)
        SR_data = model(input, target)

        #CNNR
        # loss2 = EuclideanLoss(fea1, fea2) #NOTE: confused about loss2
        # TODO
        # LFR = AngleLoss(SI_angular, newlabel)
        # LFR.backward()
        # optimizer_CNNR.step()

        #CNNH
        LSR = EuclideanLoss(SR_data, target)
        # LSI = EuclideanLoss(SI_embed_HR, SI_embed_SR.detach())
        newimg = torch.cat((SR_data, target),0)
        newimg = pjnet(newimg)
        fea1 = newimg[0:opt.batchSize, :]
        fea2 = newimg[opt.batchSize:, :]
        LSI = EuclideanLoss(fea1, fea2.detach())
        L = LSR + opt.alpha * LSI
        if opt.judgeloss != 0.0:
            L += opt.judgeloss * pjnet_loss_fn(SR_data, target, opt.batchSize)
        L.backward()
        optimizer_CNNH.step()


            # epoch_loss += output.item()
            # output.backward()
            # optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), L.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        cnt = 0
        for batch in testing_data_loader:
            cnt += 1
            input, target = batch[0].to(device), batch[1].to(device)


            # loss_num = model(input, target)
            SR_data = model(input, target)
        
            pj_loss = pjnet_loss_fn(SR_data, target, opt.batchSize)
            avg_loss = 0.0
            avg_loss += pj_loss
    print("===> Avg. IS: {:.4f} ".format(avg_loss / cnt))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    
    checkpoint(epoch)
