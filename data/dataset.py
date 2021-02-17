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


def png_saver(filename, img):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img = functional.to_pil_image(img)
    img.save(filename)


def txt_loader(filename, default=None):
    if os.path.exists(filename):
        data = np.loadtxt(filename)
        data = FloatTensor(data)
        return data
    else:
        return default


def txt_saver(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = data.detach().cpu().numpy()
    np.savetxt(filename, data)


class FolderDataset(Dataset):
    def __init__(self, root, split='train'):
        super().__init__()
        self.root = root
        self.split = split
        samples = len(glob.glob(os.path.join(self.root, 'face', '*.png')))
        if self.split == 'all':
            self.samples = range(samples)
        elif self.split == 'train':
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
        light = txt_loader(template % ('light', index, 'txt'),
                           default=torch.zeros((27, ), dtype=torch.float))
        albedo = png_loader(template % ('albedo', index, 'png'),
                            default=torch.zeros_like(face))
        normal = png_loader(template % ('normal', index, 'png'),
                            default=torch.zeros_like(face))
        return {
            'face': face,
            'mask': mask,
            'light': light,
            'albedo': albedo,
            'normal': normal,
        }

    def __setitem__(self, index, value):
        index = self.samples[index]
        template = os.path.join(self.root, '%s', '%08d.%s')
        for name, item in value.items():
            assert name != 'face'
            if name == 'light':
                txt_saver(template % (name, index, 'txt'), item)
            else:
                png_saver(template % (name, index, 'png'), item)


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
