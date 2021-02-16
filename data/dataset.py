# -*- encoding=utf-8 -*-

import os
import glob
import torch
import numpy as np
from PIL import Image
from torch import FloatTensor
from torch.utils.data import Dataset
from torchvision.transforms import functional

__all__ = ['DatasetX', 'FolderDataset']


def png_loader(filename, size=128, default=None):
    if os.path.exists(filename):
        img = Image.open(filename)
        img = functional.resize(img, size, Image.BILINEAR)
        img = functional.center_crop(img, (size, size))
        img = functional.to_tensor(img)
        return img
    else:
        return default


def txt_loader(filename):
    data = np.loadtxt(filename)
    data = FloatTensor(data)
    return data


class FolderDataset(Dataset):
    def __init__(self, root, split='train'):
        super().__init__()
        self.root = root
        self.split = split
        samples = len(glob.glob(os.path.join(self.root, 'face', '*.png')))
        if self.split == 'train':
            self.samples = range(0, int(samples * 0.8))
        else:
            self.samples = range(int(samples * 0.8), samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        index = self.samples[index]
        template = os.path.join(self.root, '%s', '%08d.%s')
        face = png_loader(template % ('face', index, 'png'))
        mask = png_loader(template % ('mask', index, 'png'),
                          default=torch.ones_like(face))
        light = txt_loader(template % ('light', index, 'txt'))
        albedo = png_loader(template % ('albedo', index, 'png'))
        normal = png_loader(template % ('normal', index, 'png'))
        return {
            'face': face,
            'mask': mask,
            'light': light,
            'albedo': albedo,
            'normal': normal,
        }


class DatasetX(Dataset):
    def __init__(self):
        super().__init__()
        self.datasets = list()

    def append(self, dataset):
        self.datasets.append(dataset)

    def __len__(self):
        return sum(map(len, self.datasets))

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            else:
                index -= len(dataset)
