import torch.utils.data as data
import cv2, sys
import numpy as np
import torch

from gen_lr import gen_lr
from os import listdir
from os.path import join
from PIL import Image
from matlab_cp2tform import get_similarity_transform_for_cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    return cv2.imdecode(np.fromfile(filepath, np.uint8), 1)

def alignment(src_img, src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def load_img_info(info_dir):
    f = open(info_dir)
    imgs = []
    for x in f.readlines():
        if (x[-1]=='\n'):
            x = x[:-1]
        y = x.split(' ')
        img_path = y[1]
        label = int(y[2])
        imgs.append([img_path, label])
    return imgs
'''
[
    [file_path, label, [landmark0, landmark1, ... , landmark9]],...
]
'''

class RecDatasetFromFolder(data.Dataset):
    # dataloader's shuffle should be false
    def __init__(self, HR_dir, LR_dir, info_dir):
        super(RecDatasetFromFolder, self).__init__()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir
        self.images_info = load_img_info(info_dir)

    def __getitem__(self, index):
        file_dir, label = self.images_info[index]
        HR = load_img(join(self.HR_dir, file_dir))
        LR = load_img(join(self.LR_dir, file_dir))
        HR = HR.transpose(2, 0, 1) # 112 * 96
        HR = (HR - 127.5) / 128.0
        LR = LR.transpose(2, 0, 1) # 28 * 24
        LR = (LR - 127.5) / 128.0
        HR = torch.from_numpy(HR).float()
        LR = torch.from_numpy(LR).float()
        return LR, HR, label
        
    def __len__(self):
        return len(self.images_info) # 20 images per folder

class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, HR_image_dir, LR_image_dir):
        super(TrainDatasetFromFolder, self).__init__()
        self.HR_image_dir = HR_image_dir
        self.LR_image_dir = LR_image_dir
        self.image_filenames = [x for x in listdir(LR_image_dir) if is_image_file(x)]

    def __getitem__(self, index):
        input = load_img(join(self.LR_image_dir, self.image_filenames[index]))
        target = load_img(join(self.HR_image_dir, self.image_filenames[index]))
        input = input.transpose(2, 0, 1) # 28 x 24
        input = (input - 127.5) / 128.0
        target = target.transpose(2, 0, 1) # 112 x 96
        target = (target - 127.5) / 128.0
        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()
        return input, target

    def __len__(self):
        return len(self.image_filenames)

class TestDatasetFromFolder(data.Dataset):
    def __init__(self, HR_image_dir, LR_image_dir):
        super(TestDatasetFromFolder, self).__init__()
        self.HR_image_dir = HR_image_dir
        self.LR_image_dir = LR_image_dir
        self.image_filenames = [x for x in listdir(LR_image_dir) if is_image_file(x)]

    def __getitem__(self, index):
        input = load_img(join(self.LR_image_dir, self.image_filenames[index]))
        target = load_img(join(self.HR_image_dir, self.image_filenames[index]))
        input = input.transpose(2, 0, 1)
        input = (input - 127.5) / 128.0
        target = target.transpose(2, 0, 1)
        target = (target - 127.5) / 128.0
        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()
        return input, target, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)
