from __future__ import print_function
import argparse, os, cv2

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from model import CNNHNet
from dataset import TrainDatasetFromFolder, TestDatasetFromFolder
from score import evaluate
import torch.nn.functional as F

import net_sphere

def get_training_set(dir):
    return TrainDatasetFromFolder(dir + '/train_HR', dir + '/train_LR')

def get_test_set(dir):
    return TestDatasetFromFolder(dir + '/valid_HR', dir + '/valid_LR')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--bs', type=int, default=256, help='training batch size')
parser.add_argument('--test_bs', type=int, default=128, help='testing batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--alpha', type=float, default=16, help='alpha to combine LSR and LSI in the paper algorithm 1')
parser.add_argument('--train', type=str, default='/home/zhaocg/celeba/dataset', help='path to training dataset')
parser.add_argument('--result', type=str, default='results', help='result dir')
parser.add_argument('--model_output', type=str, default='models', help='model output dir')
parser.add_argument('--alpha_rate', type=float, default='1', help='alpha increase rate')
parser.add_argument('--device', type=int, default=0, help='gpu device')
options = parser.parse_args()

print(options)

if not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

torch.manual_seed(options.seed)
device = torch.device('cuda')
torch.cuda.set_device(options.device)

print('[!] Loading datasets ... ', end='', flush=True)
train_set = get_training_set(options.train)
test_set = get_test_set(options.train)
train_data_loader = DataLoader(dataset=train_set, num_workers=options.threads, batch_size=options.bs, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_set, num_workers=options.threads, batch_size=options.test_bs, shuffle=False, drop_last=False)
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

# optimizer_cnn_h = optim.SGD(cnn_h.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.00025)
optimizer_cnn_h = optim.Adam(cnn_h.parameters(), lr=options.lr)
EuclideanLoss = nn.MSELoss()
# AngleLoss = net_sphere.AngleLoss()

def train(epoch):
    print('[!] Training epoch ' + str(epoch) + ' ...')
    options.alpha *= options.alpha_rate
    print(' -  Current learning rate is ' + str(options.lr), flush=True)
    print(' -  Current alpha is ' + str(options.alpha), flush=True)
    bs = options.bs
    for iteration, batch in enumerate(train_data_loader):
        input, target = batch[0].to(device), batch[1].to(device)
        optimizer_cnn_h.zero_grad()

        sr_img = cnn_h(input)
        l_sr = EuclideanLoss(sr_img, target)

        features = cnn_r(torch.cat((sr_img, target), 0))
        f1 = features[0:bs, :]; f2 = features[bs:, :]
        f1_norm = F.normalize(f1); f2_norm = F.normalize(f2)
        l_si = EuclideanLoss(f1_norm, f2_norm.detach())
        loss = l_sr + options.alpha * l_si
        loss.backward()
        optimizer_cnn_h.step()

        print(' -  Epoch[{}] ({}/{}): Loss: {:.4f} sr: {:.4f} si: {:.8f}\r'.format(epoch, iteration, len(train_data_loader), loss.item(), l_sr.item(), l_si.item()), end='')

    print('\n[!] Epoch {} complete.'.format(epoch))

def output_img(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with torch.no_grad():
        for batch in test_data_loader:
            input, target, filename = batch[0].to(device), batch[1].to(device), batch[2]
            sr = cnn_h(input).cpu()
            for i in range(len(filename)):
                img = sr[i] * 128 + 127.5
                img = img.numpy().transpose(1, 2, 0)
                cv2.imwrite(output_dir + '/' + filename[i].split('/')[-1], img)

def test_and_save(epoch):
    print('[!] Saving test results ... ', flush=True, end='')
    dir_name = options.result + '/output_' + str(epoch) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_img(dir_name)
    print('done !', flush=True)
    evaluate(dir_name, options.train + '/valid_HR', options.train + '/valid_LR', cnn_r)

def checkpoint(epoch):
    model_out_path = options.model_output + '/model_epoch_{}'.format(epoch) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pth'
    print('[!] Saving checkpoint into ' + model_out_path + ' ... ', flush=True, end='')
    torch.save(cnn_h, model_out_path)
    print('done !', flush=True)

for epoch in range(1, options.epochs + 1):
    train(epoch)
    test_and_save(epoch)
    checkpoint(epoch)
