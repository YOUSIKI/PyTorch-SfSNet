# -*- coding: utf-8 -*-

import os
import h5py
import glob
import math
import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from random import sample


class SyntheticFace(torch.utils.data.Dataset):
    def __init__(self, root, train, image_size=128, random_scale=1):
        super().__init__()
        self.image_size = image_size
        templates = list(
            map(lambda s: s.replace('face', '{name}').replace('png', '{ext}'),
                sorted(glob.glob(os.path.join(root, '*/*_face_*.png')))))
        if train:
            templates = templates[:int(len(templates) * 0.8)]
        else:
            templates = templates[int(len(templates) * 0.8):]
        if random_scale != 1:
            templates = sample(templates,
                               math.ceil(random_scale * len(templates)))
        self.templates = templates

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, index):
        template = self.templates[index]
        returns = dict()
        for name in 'face mask depth albedo normal light'.split():
            if name == 'light':
                returns[name] = torch.FloatTensor(
                    np.loadtxt(template.format(
                        name=name,
                        ext='txt',
                    )))
            else:
                img = Image.open(template.format(name=name, ext='png'))
                img = F.resize(img, self.image_size)
                img = F.to_tensor(img)
                if name == 'normal':
                    img = (img / 128.0) - 1.0
                returns[name] = img
        return returns


class SyntheticFace2(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 train,
                 names=['face', 'mask', 'depth', 'albedo', 'normal', 'light']):
        super().__init__()
        self.file = h5py.File(path, 'r')
        self.train = train
        self.names = names
        self.indices = list(range(len(self.file['face'])))
        if train:
            self.indices = self.indices[:int(len(self.indices) * 0.8)]
        else:
            self.indices = self.indices[int(len(self.indices) * 0.8):]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        return {
            name: self.transform(self.file[name][index])
            for name in self.names
        }

    @staticmethod
    def transform(x: np.ndarray):
        if x.ndim == 3:  # image
            return F.to_tensor(x)
        else:
            return torch.FloatTensor(x)
