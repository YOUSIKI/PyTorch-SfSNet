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


def png_loader(filename, size=128):
    img = Image.open(filename)
    img = functional.resize(img, size, Image.BILINEAR)
    img = functional.center_crop(img, (size, size))
    img = functional.to_tensor(img)
    return img


def txt_loader(filename):
    data = np.loadtxt(filename)
    data = FloatTensor(data)
    return data


class FolderDataset(Dataset):
    def __init__(self,
                 root,
                 split='train',
                 device=torch.device('cpu'),
                 teacher=None):
        super().__init__()
        self.root = root
        self.split = split
        self.device = device
        self.teacher = teacher
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
        face = png_loader(template % ('face', index, 'png')).to(self.device)
        if self.teacher is None:
            mask = png_loader(template % ('mask', index, 'png'))
            light = txt_loader(template % ('light', index, 'txt'))
            albedo = png_loader(template % ('albedo', index, 'png'))
            normal = png_loader(template % ('normal', index, 'png'))
        else:
            mask = torch.ones_like(face)
            with torch.no_grad():
                self.teacher = self.teacher.to(self.device)
                self.teacher.eval()
                outputs = self.teacher({'face': face.view((1, *face.size()))})
                light = outputs['light'].detach()
                albedo = outputs['albedo'].detach()
                normal = outputs['normal'].detach()
                light = light.view(light.size()[1:])
                albedo = albedo.view(albedo.size()[1:])
                normal = normal.view(normal.size()[1:])
        mask = mask.to(self.device)
        light = light.to(self.device)
        albedo = albedo.to(self.device)
        normal = normal.to(self.device)
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
