import os, cv2
import torch
import argparse

import numpy as np
import net_sphere
from torch.autograd import Variable

def evaluate(SR_dir, HR_dir, LR_dir, net, cuda=True):
    print('[!] Evaluating results ... ', end='', flush=True)
    sum = 0; min_v = 2; max_v = -1; count = 0
    sum_bicubic = 0; min_v_bicubic = 2; max_v_bicubic = -1
    score = 0;
    for i in os.listdir(HR_dir):
        file_HR = os.path.join(HR_dir, i)
        file_SR = os.path.join(SR_dir, i)
        file_LR = os.path.join(LR_dir, i)
        img_HR = cv2.imdecode(np.fromfile(file_HR, np.uint8), 1)
        img_SR = cv2.imdecode(np.fromfile(file_SR, np.uint8), 1)
        img_LR = cv2.imdecode(np.fromfile(file_LR, np.uint8), 1)
        img_LR = cv2.resize(img_LR, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC) # bicubic
        img_list = [img_HR, cv2.flip(img_HR, 1), img_SR, cv2.flip(img_SR, 1), img_LR, cv2.flip(img_LR, 1)]
        for i in range(len(img_list)):
            img_list[i] = img_list[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            img_list[i] = (img_list[i] - 127.5) / 128.0
        img = np.vstack(img_list)
        if cuda:
            with torch.no_grad():
                img = Variable(torch.from_numpy(img).float()).cuda()
        else:
            with torch.no_grad():
                img = Variable(torch.from_numpy(img).float())
        output = net(img)
        f = output.data
        f1, f2, f3 = f[0], f[2], f[4]
        cos_distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        cos_distance_bicubic = f1.dot(f3) / (f1.norm() * f3.norm() + 1e-5)
        cos_distance = float(cos_distance.float()); cos_distance_bicubic = float(cos_distance_bicubic.float())
        sum += cos_distance; sum_bicubic += cos_distance_bicubic;
        if min_v > cos_distance: min_v = cos_distance
        if max_v < cos_distance: max_v = cos_distance
        if min_v_bicubic > cos_distance_bicubic: min_v_bicubic = cos_distance_bicubic
        if max_v_bicubic < cos_distance_bicubic: max_v_bicubic = cos_distance_bicubic
        count += 1
        score += ((cos_distance - cos_distance_bicubic) / (1.0 - cos_distance_bicubic)) ** 2 # assume best is perfect
    print('done !')
    print(' -  ave: ' + str(sum / count) + ' / ' + str(sum_bicubic / count))
    print(' -  min: ' + str(min_v) + ' / ' + str(min_v_bicubic))
    print(' -  max: ' + str(max_v) + ' / ' + str(max_v_bicubic))
    print(' -  score: ' + str(18.0 * score / count), flush=True)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple scoring script')
    parser.add_argument('--hr', type=str, default='')
    parser.add_argument('--sr', type=str, default='')
    parser.add_argument('--lr', type=str, default='')
    parser.add_argument('--net', type=str, default='sphere20a')
    parser.add_argument('--model', default='sphere20a.pth', type=str)
    parser.add_argument('--cuda', default='yes', type=str)
    options = parser.parse_args()

    cuda = (options.cuda == 'yes')
    net = getattr(net_sphere, options.net)()
    net.load_state_dict(torch.load(options.model))
    if cuda:
        net.cuda()
    net.eval()
    net.feature = True
    evaluate(options.sr, options.hr, options.lr, net, cuda)