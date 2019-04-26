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

def get_training_set(dir, options):
    return TrainDatasetFromFolder(dir + '/train_HR', dir + '/train_LR', options)

def get_test_set(dir, options):
    return TestDatasetFromFolder(dir + '/valid_HR', dir + '/valid_LR', options)

def get_final_set(dir, options):
    return TestDatasetFromFolder(dir, dir, options)






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
parser.add_argument('--train', type=str, default='/home/huangkz/data/celeba/dataset', help='path to training dataset', required=True)
parser.add_argument('--result', type=str, default='results', help='result dir')
parser.add_argument('--model_output', type=str, default='model', help='model output dir')
parser.add_argument('--alpha_rate', type=float, default='1', help='alpha increase rate')
parser.add_argument('--device', type=int, default=0, help='gpu device')
parser.add_argument('--load', type=str, default='', help='load model')
parser.add_argument('--only_test', action='store_true', default=False, help='only test mode')
parser.add_argument('--final_test_dir', default='/home/huangkz/data/testset/', type=str, help='infer the valid data and the final test,'
                                                    ' using all models in preparedmodels by default', required=True)
parser.add_argument('--final_output_dir', default='/home/huangkz/data/finaloutput/wavelettrain', type=str, help='infer the valid data and the final test,', required=True)

parser.add_argument('--train_data_cut', type=float, default=1.0, help="trainning data cut")
parser.add_argument('--test_data_cut', type=float, default=1.0, help="testting data cut")

options = parser.parse_args()

print(options)

if not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

torch.manual_seed(options.seed)
device = torch.device('cuda')
torch.cuda.set_device(options.device)

print('[!] Loading datasets ... ', end='', flush=True)
if not options.only_test:
    train_set = get_training_set(options.train, options)
    train_data_loader = DataLoader(dataset=train_set, num_workers=options.threads, batch_size=options.bs, shuffle=True, drop_last=True)

test_set = get_test_set(options.train, options)
test_data_loader = DataLoader(dataset=test_set, num_workers=options.threads, batch_size=options.test_bs, shuffle=False, drop_last=False)


final_set = get_final_set(options.final_test_dir, options)
final_data_loader = DataLoader(dataset=final_set, num_workers=4, batch_size=options.test_bs, shuffle=False, drop_last=False)

print('done !', flush=True)

print('[!] Building model ... ', end='', flush=True)
cnn_h = CNNHNet(upscale_factor=options.upscale_factor, batch_size=options.bs)
cnn_h = nn.DataParallel(cnn_h)
if options.load:
    cnn_h.load_state_dict(torch.load(options.load))
cnn_r = getattr(net_sphere, 'sphere20a')()
cnn_r.load_state_dict(torch.load('sphere20a.pth'))
cnn_r.feature = True

for param in cnn_r.parameters():
    param.requires_grad = False
cnn_h = cnn_h.cuda()
cnn_r = nn.DataParallel(cnn_r).cuda()
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

def output_final(score):
    output_dir = options.final_output_dir + '_traingyx_' + score + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with torch.no_grad():
        for batch in final_data_loader:
            input, target, filename = batch[0].to(device), batch[1].to(device), batch[2]
            sr = cnn_h(input).cpu()
            for i in range(len(filename)):
                img = sr[i] * 128 + 127.5
                img = img.numpy().transpose(1, 2, 0)
                cv2.imwrite(output_dir + '/' + filename[i].split('/')[-1], img)


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def test_and_save(epoch):
    print('[!] Saving test results ... ', flush=True, end='')
    dir_name = options.result + '/gyxnet_output_@_' + str(epoch) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_img(dir_name)
    print('done !', flush=True)
    finalscore = evaluate(dir_name, options.train + '/valid_HR', options.train + '/valid_LR', cnn_r)
    os.rename(dir_name, dir_name.replace('@', finalscore))
    output_final(finalscore)
    return finalscore

def checkpoint(epoch, score):
    model_out_path = options.model_output + '/model_'+score + '_epoch_{}'.format(epoch) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pth'
    print('[!] Saving checkpoint into ' + model_out_path + ' ... ', flush=True, end='')
    save_model(cnn_h, model_out_path)
    print('done !', flush=True)

if options.only_test:
    test_and_save(0)
else:
    for epoch in range(1, options.epochs + 1):
        train(epoch)
        score = test_and_save(epoch)
        checkpoint(epoch, score)
