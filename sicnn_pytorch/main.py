from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNNHNet
from dataset import DatasetFromFolder

import net_sphere

def get_training_set(dir):
    return DatasetFromFolder(dir + '/train_HR', dir + '/train_LR')

def get_test_set(dir):
    return DatasetFromFolder(dir + '/valid_HR', dir + '/valid_LR')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--bs', type=int, default=64, help='training batch size')
parser.add_argument('--test_bs', type=int, default=64, help='testing batch size')# todo
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--alpha', type=float, default=10.0, help='alpha to combine LSR and LSI in the paper algorithm 1')
parser.add_argument('--train', type=str, default='/home/zhaocg/celeba/dataset', help='path to training dataset')
options = parser.parse_args()

print(options)

if not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

torch.manual_seed(options.seed)
device = torch.device('cuda')

print('[!] Loading datasets ... ', end='', flush=True)
train_set = get_training_set(options.train)
test_set = get_test_set(options.train)
train_data_loader = DataLoader(dataset=train_set, num_workers=options.threads, batch_size=options.bs, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_set, num_workers=options.threads, batch_size=options.test_bs, shuffle=False, drop_last=True)
print('done !', flush=True)

print('[!] Building model ... ', end='', flush=True)
cnn_h = CNNHNet(upscale_factor=options.upscale_factor, batch_size=options.bs).to(device)
cnn_r = getattr(net_sphere, 'sphere20a')()
cnn_r.load_state_dict(torch.load('sphere20a.pth'))
cnn_r.feature = True

for param in cnn_r.parameters():
    param.requires_grad = False
cnn_r = cnn_r.cuda()
print('done !', flush=True)

optimizer_cnn_h = optim.Adam(cnn_h.parameters(), lr=options.lr)
EuclideanLoss = nn.MSELoss()
AngleLoss = net_sphere.AngleLoss()

def train(epoch):
    print('[!] Training epoch ' + str(epoch) + ' ...')
    bs = options.bs
    for iteration, batch in enumerate(train_data_loader):
        if iteration > 200:
            break
        input, target = batch[0].to(device), batch[1].to(device)
        optimizer_cnn_h.zero_grad()

        sr_img = cnn_h(input)
        l_sr = EuclideanLoss(sr_img, target)

        features = cnn_r(torch.cat((sr_img, target), 0))
        f1 = features[0:bs, :]; f2 = features[bs:, :]
        l_si = EuclideanLoss(f1, f2.detach())
        loss = l_sr + options.alpha * l_si
        loss.backward()
        optimizer_cnn_h.step()

        print(' -  Epoch[{}] ({}/{}): Loss: {:.4f}'.format(epoch, iteration, len(train_data_loader), loss.item()))

    print('[!] Epoch {} complete.'.format(epoch))


def test_and_save():
    pass

def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    print('[!] Saving checkpoint into ' + model_out_path + ' ... ', flush=True, end='')
    torch.save(model, model_out_path)
    print('done !', flush=True)

for epoch in range(1, options.epochs + 1):
    train(epoch)
    test_and_save()
    checkpoint(epoch)
