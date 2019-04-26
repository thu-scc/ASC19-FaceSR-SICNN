from PIL import Image, ImageDraw
import process_aligned

import argparse
import cv2
import numpy as np
import os
import torch
import GEN_LR
from model import CNNHNet
import torch
import torch.nn as nn

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face SR, Tsinghua University, ASC 19')
    parser.add_argument('--input', type=str, default='or', help='Input directory')
    parser.add_argument('--output_cp', type=str, default='cp', help='Output directory')
    parser.add_argument('--output_lr', type=str, default='lr', help='Output directory')
    parser.add_argument('--output_sr', type=str, default='sr', help='Output directory')
    options = parser.parse_args()

    print('[!] Cropping images ... ', flush=True, end='')
    process_aligned.process(options.input, options.output_cp)
    print('done !')

    print('[!] Generating LR ... ', flush=True, end='')
    GEN_LR.process(options.output_cp, options.output_lr)
    print('done !')

    print('[!] Generating SR ... ', flush=True, end='')
    cnn_h = CNNHNet(upscale_factor=4, batch_size=1)
    cnn_h = nn.DataParallel(cnn_h)
    cnn_h.load_state_dict(torch.load('sicnn.pth'))
    for param in cnn_h.parameters():
        param.requires_grad = False
    cnn_h = cnn_h.cpu()
    with torch.no_grad():
        for file in os.listdir(options.output_lr):
            if not is_image_file(file.lower()): continue
            input = cv2.imread(os.path.join(options.output_lr, file), 1)
            input = input.transpose(2, 0, 1)
            input = (input - 127.5) / 128.0
            input = torch.from_numpy(input.reshape((1, ) + input.shape)).float()
            sr = cnn_h(input).cpu()
            img = sr[0] * 128 + 127.5
            img = img.numpy().transpose(1, 2, 0)
            cv2.imwrite(os.path.join(options.output_sr, file), img)
    print('done !')