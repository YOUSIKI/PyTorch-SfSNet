# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class SfSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_features = self.BaseFeatures()
        self.albedo_features = self.ResidualBlocks()
        self.normal_features = self.ResidualBlocks()
        self.albedo_generator = self.GeneratationBlocks()
        self.normal_generator = self.GeneratationBlocks()
        self.light_estimator = self.LightEstimator()

    def forward(self, face):
        features = self.base_features(face)
        albedo_features = self.albedo_features(features)
        normal_features = self.normal_features(features)
        albedo = self.albedo_generator(albedo_features)
        normal = self.normal_generator(normal_features)
        light = self.light_estimator(
            torch.cat([
                features,
                albedo_features,
                normal_features,
            ], dim=1))
        shading = get_shading(normal, light)
        face = shading * albedo
        return {
            'face': face,
            'albedo': albedo,
            'normal': normal,
            'light': light,
            'shading': shading
        }

    @staticmethod
    def BaseFeatures():
        return nn.Sequential(
            Conv2d(3, 64, kernel_size=7, norm='bn', act='relu'),
            Conv2d(64, 128, kernel_size=3, norm='bn', act='relu'),
            Conv2d(128, 128, kernel_size=3, stride=2),
        )

    @staticmethod
    def ResidualBlocks(n_blocks=5):
        return nn.Sequential(
            *[
                Residual(
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                ) for i in range(n_blocks)
            ],
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def GeneratationBlocks():
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2d(128, 128, kernel_size=1, norm='bn', act='relu'),
            Conv2d(128, 64, kernel_size=3, norm='bn', act='relu'),
            Conv2d(64, 3, kernel_size=1),
        )

    @staticmethod
    def LightEstimator():
        return nn.Sequential(
            Conv2d(128 * 3, 128, kernel_size=1, norm='bn', act='relu'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 27),
        )


@torch.jit.script
def get_shading(N, L):
    device = N.device
    c1 = 0.8862269254527579
    c2 = 1.0233267079464883
    c3 = 0.24770795610037571
    c4 = 0.8580855308097834
    c5 = 0.4290427654048917
    nx = N[:, 0, :, :]
    ny = N[:, 1, :, :]
    nz = N[:, 2, :, :]
    b, c, h, w = N.size()
    Y1 = c1 * torch.ones((b, h, w), device=device)
    Y2 = c2 * nz
    Y3 = c2 * nx
    Y4 = c2 * ny
    Y5 = c3 * (2 * nz * nz - nx * nx - ny * ny)
    Y6 = c4 * nx * nz
    Y7 = c4 * ny * nz
    Y8 = c5 * (nx * nx - ny * ny)
    Y9 = c4 * nx * ny
    sh = torch.split(L, 9, dim=1)
    shading = torch.zeros((b, c, h, w), device=device)
    for j, l in enumerate(sh):
        l = l.repeat(1, h * w).view(b, h, w, 9).permute([0, 3, 1, 2])
        shading[:, j, ...] = \
            Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
            Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
            Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]
    return shading
