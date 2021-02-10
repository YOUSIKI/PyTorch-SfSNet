# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils
from ignite.engine import Engine, Events

import models
import datasets


def parse_arguments():
    parser = argparse.ArgumentParser('SfSNet Evaluating')
    parser.add_argument('--root_synthetic',
                        type=str,
                        default=None,
                        action='store')
    parser.add_argument('--dir_samples',
                        type=str,
                        default='samples',
                        action='store')
    parser.add_argument('--num_samples', type=int, default=8, action='store')
    parser.add_argument('--pretrained', type=str, default=None, action='store')
    parser.add_argument('--image_size', type=int, default=128, action='store')
    parser.add_argument('--num_workers', type=int, default=2, action='store')
    parser.add_argument('--apply_mask', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    print('Parsed Arguments:', args)
    return args


def main(args):
    valid_ds = datasets.SyntheticFace(
        args.root_synthetic,
        train=False,
        image_size=args.image_size,
    )
    valid_ds = [valid_ds[i] for i in range(args.num_samples)]
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
    )

    if args.cuda and not torch.cuda.is_available():
        raise ValueError('CUDA support is not available, disabled')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    net = models.SfSNet()

    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained))

    net = net.to(device)

    def valid_step(engine, batch):
        net.eval()
        with torch.no_grad():
            target = {
                k: v.to(device)
                for k, v in batch.items()
                if k in 'face albedo normal light'.split()
            }
            mask = batch['mask'].to(device)
            if args.apply_mask:
                target = {
                    k: v if k == 'light' else v * mask
                    for k, v in target.items()
                }
            output = net(target['face'])
            if args.apply_mask:
                output = {
                    k: v if k == 'light' else v * mask
                    for k, v in output.items()
                }
        dirpath = os.path.join(args.dir_samples, str(engine.state.iteration))
        os.makedirs(dirpath, exist_ok=True)
        for name in 'face albedo'.split():
            tvutils.save_image(target[name],
                               os.path.join(dirpath, f'{name}_target.png'))
            tvutils.save_image(output[name],
                               os.path.join(dirpath, f'{name}_output.png'))
        # normal
        tvutils.save_image((target['normal'] + 1.0) * 128.0,
                           os.path.join(dirpath, f'normal_target.png'))
        tvutils.save_image((output['normal'] + 1.0) * 128.0,
                           os.path.join(dirpath, f'normal_output.png'))
        # light
        print(torch.abs(target['light'] - output['light']).cpu())

    evaluator = Engine(valid_step)

    evaluator.run(valid_dl)


if __name__ == '__main__':
    main(parse_arguments())