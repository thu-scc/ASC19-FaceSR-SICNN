from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from dataset import DatasetFromFolder, TrainDatasetFromFolder

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform(crop_size, upscale_factor):
    return Compose([ CenterCrop(crop_size), Resize(crop_size // upscale_factor), ToTensor()])

def target_transform(crop_size):
    return Compose([CenterCrop(crop_size), ToTensor()])

def normal_transform():
    return Compose([ToTensor()])

def get_training_set(dir):
    return TrainDatasetFromFolder(dir + '/train_HR', dir + '/train_LR', device_transform=normal_transform(), target_transform=normal_transform())

def get_test_set(dir):
    return TrainDatasetFromFolder(dir + '/valid_HR', dir + '/valid_LR', device_transform=normal_transform(), target_transform=normal_transform())
