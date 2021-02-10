# -*- coding: utf-8 -*-

import os
import torch
from PIL import Image
from glob import glob
from numpy import loadtxt
from torchvision import transforms


class SyntheticFace(torch.utils.data.Dataset):
    def __init__(self, root, train, image_size=128):
        super().__init__()
        self.root = root
        self.train = train
        self.image_size = image_size
        self.templates = list(
            map(lambda s: s.replace('face', '{part}').replace('png', '{ext}'),
                sorted(glob(os.path.join(root, '*/*_face_*.png')))))
        if self.train:
            self.templates = self.templates[:int(len(self.templates) * 0.9)]
        else:
            self.templates = self.templates[int(len(self.templates) * 0.9):]
        base_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.albedo_transform = base_transform
        self.depth_transform = base_transform
        self.face_transform = base_transform
        self.mask_transform = base_transform
        self.normal_transform = transforms.Compose([
            base_transform,
            transforms.Lambda(lambda i: (i - 128.0) / 128.0),
        ])

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, index):
        template = self.templates[index]
        return {
            part: getattr(self, '{}_transform'.format(part))(Image.open(
                template.format(part=part, ext='png')))
            for part in 'face albedo normal depth mask'.split()
        } | {
            'light':
            torch.tensor(loadtxt(template.format(part='light', ext='txt')),
                         dtype=torch.float)
        }
