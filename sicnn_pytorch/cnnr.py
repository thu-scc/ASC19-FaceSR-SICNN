from __future__ import print_function
import argparse, os, cv2

import torch
import torch.nn as nn
import torch.optim as optim
import time
import net_sphere

from torch.utils.data import DataLoader
from model import CNNHNet
from dataset import TestDatasetFromFolder, RecDatasetFromFolder
from score import evaluate

def get_train_set(dataset_dir, mapping):
    return RecDatasetFromFolder(dataset_dir + '/HR', dataset_dir + '/LR', mapping)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--bs', type=int, default=256, help='training batch size')
parser.add_argument('--test_bs', type=int, default=256, help='testing batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr_cnnr', type=float, default=0.01, help='Learning Rate. Default=0.1')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--train', type=str, default='/home/heheda/casia', help='path to cnnr dataset')
parser.add_argument('--label', type=str, default='/home/heheda/casia/mapping.txt', help='path to training dataset')
parser.add_argument('--model_output', type=str, default='models', help='model output dir')
options = parser.parse_args()

print(options)

if not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

device = torch.device('cuda')

print('[!] Loading datasets ... ', end='', flush=True)
train_set = get_train_set(options.train, options.label)
train_data_loader = DataLoader(dataset=train_set, num_workers=options.threads, batch_size=options.bs, shuffle=False, drop_last=True)
print('done !', flush=True)

print('[!] Building model ... ', end='', flush=True)
cnn_r_train = getattr(net_sphere, 'sphere20a')()
cnn_r_train = cnn_r_train.cuda()
print('done !', flush=True)

AngleLoss = net_sphere.AngleLoss()

def train(epoch):
    print('[!] Training epoch ' + str(epoch) + ' ...')
    print(' -  Current learning rate is ' + str(options.lr_cnnr), flush=True)

    for iteration, batch in enumerate(train_data_loader):
        hr, labels = batch[1].cuda(), batch[2].cuda()
        optimizer_cnn_r_train.zero_grad()

        output_labels = cnn_r_train(hr)
        loss = AngleLoss(output_labels, labels)
        loss.backward()
        optimizer_cnn_r_train.step()
        print(' -  Epoch[{}] ({}/{}): Loss: {:.4f}\r'.format(epoch, iteration, len(train_data_loader), loss.item()), end='')

    print('[!] Epoch {} complete.'.format(epoch))

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def checkpoint(epoch):
    cnn_r_out_path = options.model_output + '/cnn_r_epoch_{}'.format(epoch) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pth'
    print('[!] Saving checkpoint into ' + cnn_r_out_path + ' ... ', flush=True, end='')
    save_model(cnn_r_train, cnn_r_out_path)

    print('done !', flush=True)

options.lr_cnnr *= 10
for epoch in range(1, options.epochs + 1):
    if epoch in [1, 11, 16, 19]:
        options.lr_cnnr *= 0.1
        optimizer_cnn_r_train = optim.SGD(cnn_r_train.parameters(), lr=options.lr_cnnr, momentum=0.9, weight_decay=5e-4)
    train(epoch)
    checkpoint(epoch)
