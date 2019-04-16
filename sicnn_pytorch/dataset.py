import torch.utils.data as data
import cv2, sys
import numpy as np
import torch

from os import listdir
from os.path import join
from PIL import Image
from matlab_cp2tform import get_similarity_transform_for_cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    return cv2.imdecode(np.fromfile(filepath, np.uint8), 1)

def load_img_info(mapping):
    f = open(mapping)
    imgs = []
    for x in f.readlines():
        if x[-1] == '\n':
            x = x[:-1]
        y = x.split(' ')
        img_path = y[1]
        label = int(y[2])
        imgs.append([img_path, label])
    return imgs

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

class RecDatasetFromFolder(data.Dataset):
    def __init__(self, HR_dir, LR_dir, mapping):
        super(RecDatasetFromFolder, self).__init__()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir
        self.images_info = load_img_info(mapping)

    def __getitem__(self, index):
        path, classid = self.images_info[index]
        HR = load_img(join(self.HR_dir, path))
        LR = load_img(join(self.LR_dir, path))
        HR = HR.transpose(2, 0, 1) # 112 * 96
        HR = (HR - 127.5) / 128.0
        LR = LR.transpose(2, 0, 1) # 28 * 24
        LR = (LR - 127.5) / 128.0
        HR = torch.from_numpy(HR).float()
        LR = torch.from_numpy(LR).float()
        label = np.zeros((1),np.float32)
        label[0] = classid
        label = torch.from_numpy(label).long()
        return LR, HR, label
        
    def __len__(self):
        return len(self.images_info)

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
