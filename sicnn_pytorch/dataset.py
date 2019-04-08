import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    # print(type(img))
    # y, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()

        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        print(input.shape)
        if self.target_transform:
            target = self.target_transform(target)
        # print(target.shape)

        input = input.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        input = (input - 127.5) / 128.0

        target = target.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        target = (target - 127.5) / 128.0

        return input, target

    def __len__(self):
        return len(self.image_filenames)



class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, HR_image_dir, LR_image_dir, device_transform=None, target_transform=None):
        super(TrainDatasetFromFolder, self).__init__()
        self.HR_image_filenames = [join(HR_image_dir, x) for x in listdir(HR_image_dir) if is_image_file(x)]
        self.LR_image_filenames = [join(LR_image_dir, x) for x in listdir(LR_image_dir) if is_image_file(x)]
        self.input_transform = device_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.LR_image_filenames[index])
        target = load_img(self.HR_image_filenames[index])
        input = self.input_transform(input)
        target = self.input_transform(target)


        return input, target

    def __len__(self):
        return len(self.HR_image_filenames)
