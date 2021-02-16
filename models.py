# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class SfSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.baseE = BaseEncoder()
        self.albedoE = Encoder()
        self.normalE = Encoder()
        self.albedoG = Generator()
        self.normalG = Generator()
        self.light = LightEstimator()
        self.shading = ShadingCalculator()

    def forward(self, inputs, fake_forward=False):
        face = inputs['face']
        if fake_forward:
            albedo = inputs['albedo']
            normal = inputs['normal']
            light = inputs['light']
        else:
            features = self.baseE(face)
            albedo_features = self.albedoE(features)
            normal_features = self.normalE(features)
            albedo = self.albedoG(albedo_features)
            normal = self.normalG(normal_features)
            features = torch.cat(
                [
                    features,
                    albedo_features,
                    normal_features,
                ],
                dim=1,
            )
            light = self.light(features)
        shading = self.shading(normal, light)
        reconstructed_face = shading * albedo
        return {
            'face': reconstructed_face,
            'albedo': albedo,
            'normal': normal,
            'light': light,
            'shading': shading,
        }


class BaseEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(*[
            Residual(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ) for i in range(5)
        ])


class Generator(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
        )


class LightEstimator(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(128 * 3, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 27),
        )


class Residual(nn.Sequential):
    def forward(self, x):
        return x + super().forward(x)


class ShadingCalculator(nn.Module):
    # TODO: rewrite this function
    def forward(self, normal, light):
        device = normal.device
        c1 = 0.8862269254527579
        c2 = 1.0233267079464883
        c3 = 0.24770795610037571
        c4 = 0.8580855308097834
        c5 = 0.4290427654048917
        nx = normal[:, 0, :, :]
        ny = normal[:, 1, :, :]
        nz = normal[:, 2, :, :]
        b, c, h, w = normal.size()
        Y1 = c1 * torch.ones((b, h, w), device=device)
        Y2 = c2 * nz
        Y3 = c2 * nx
        Y4 = c2 * ny
        Y5 = c3 * (2 * nz * nz - nx * nx - ny * ny)
        Y6 = c4 * nx * nz
        Y7 = c4 * ny * nz
        Y8 = c5 * (nx * nx - ny * ny)
        Y9 = c4 * nx * ny
        sh = torch.split(light, 9, dim=1)
        shading = torch.zeros((b, c, h, w), device=device)
        for j, l in enumerate(sh):
            l = l.repeat(1, h * w).view(b, h, w, 9).permute([0, 3, 1, 2])
            shading[:, j, ...] = \
                Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]
        return shading
