import torch.utils.data as data
import cv2, sys
import numpy as np
import torch

from os import listdir
from os.path import join
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    return cv2.imdecode(np.fromfile(filepath, np.uint8), 1)

class RecDatasetFromFolder(data.Dataset):
    # dataloader's shuffle should be false
    def __init__(self, HR_image_dir, LR_image_dir, classnum):
        super(RecDatasetFromFolder, self).__init__()
        self.HR_image_dir = HR_image_dir
        self.LR_image_dir = LR_image_dir
        self.classnum = classnum

    def __getitem__(self, index):
        label = index % self.classnum
        file_name = "%06d.jpg" % (index // self.classnum)
        folder_name = "%04d" % (index % self.classnum)
        input = load_img(join(self.LR_image_dir, folder_name, file_name))
        target = load_img(join(self.HR_image_dir, folder_name, file_name))
        input = input.transpose(2, 0, 1) # 28 x 24
        input = (input - 127.5) / 128.0
        target = target.transpose(2, 0, 1) # 112 x 96
        target = (target - 127.5) / 128.0
        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()
        return input, target, label
        
    def __len__(self):
        return self.classnum*20 # 20 images per folder

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
