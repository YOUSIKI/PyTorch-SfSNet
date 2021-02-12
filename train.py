# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from prefetch_generator import BackgroundGenerator

import models
import datasets

torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser('SfSNet Training')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        action='store',
                        help='training epochs (default: 5)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        action='store',
                        help='dataloader batch size (default: 8)')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        action='store',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--synthetic_path',
                        type=str,
                        default=None,
                        action='store',
                        help='path to SyntheticFace dataset')
    parser.add_argument('--checkpoints',
                        type=str,
                        default='ckpts',
                        action='store',
                        help='directory to save checkpoints (default: ckpts)')
    parser.add_argument('--checkpoint_interval',
                        type=int,
                        default=100,
                        action='store',
                        help='saving interval (/iterations) (default: 100)')
    parser.add_argument('--dir_samples',
                        type=str,
                        default='samples',
                        action='store',
                        help='directory to save samples (default: samples)')
    parser.add_argument('--num_samples',
                        type=str,
                        default=4,
                        action='store',
                        help='number of samples to dump (default: 4)')
    parser.add_argument('--sample_interval',
                        type=int,
                        default=100,
                        action='store',
                        help='saving samples interval (default: 100)')
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        action='store',
                        help='load pretrained weight')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        action='store',
                        help='dataloader num_workers (default: 4)')
    parser.add_argument('--apply_mask',
                        action='store_true',
                        help='apply mask on images (default: False)')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enable CUDA (default: False)')
    args = parser.parse_args()
    print('Parsed Arguments:', args)
    return args


def main(args):
    # device: cpu / cuda
    if args.cuda and not torch.cuda.is_available():
        raise ValueError('CUDA support is not available, disabled')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    # dataloaders
    class DataLoaderX(torch.utils.data.DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())

    train_dl = DataLoaderX(
        datasets.SyntheticFace2(
            args.synthetic_path,
            train=True,
            names=['face', 'albedo', 'normal', 'light', 'mask']),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    eval_dl = DataLoaderX(
        datasets.SyntheticFace2(
            args.synthetic_path,
            train=False,
            names=['face', 'albedo', 'normal', 'light', 'mask']),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # build model and load pretrained weights
    net = models.SfSNet()

    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

    net = net.to(device)

    # optimizer and learning_rate scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.2,
        verbose=True,
    )

    # universal step
    def universal_step(is_train):
        def step(engine, input):
            if is_train:
                net.train()
            else:
                net.eval()
            with (torch.enable_grad() if is_train else torch.no_grad()):
                input = {k: v.to(device) for k, v in input.items()}

                def apply_mask(x):
                    return x if x.dim() != 4 else x * input['mask']

                if args.apply_mask:
                    input = {k: apply_mask(v) for k, v in input.items()}
                output = net(input['face'])
                if args.apply_mask:
                    output = {k: apply_mask(v) for k, v in output.items()}
                loss = {
                    f'{k}_loss': F.mse_loss(v, input[k])
                    for k, v in output.items() if k in input
                }
                if is_train:
                    optimizer.zero_grad()
                    sum(loss.values()).backward()
                    optimizer.step()
                input = {f'{k}_input': v for k, v in input.items()}
                output = {f'{k}_output': v for k, v in output.items()}
                return input | output | loss

        return step

    trainer = Engine(universal_step(is_train=True))
    evaluator = Engine(universal_step(is_train=False))

    # call evaluator after every training epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run(eval_dl)

    # scheduler adjust learning_rate
    @trainer.on(Events.EPOCH_COMPLETED)
    def scheduler_step(engine):
        scheduler.step()

    # checkpoint handler
    checkpoint_handler = ModelCheckpoint(
        args.checkpoints,
        'SfSNet',
        n_saved=10,
        atomic=True,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True,
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=args.checkpoint_interval))
    def save_checkpoint(engine):
        checkpoint_handler(engine, to_save={'net': net})

    @trainer.on(Events.EXCEPTION_RAISED)
    def save_checkpoint_on_excetion(engine, e):
        engine.terminate()
        save_checkpoint(engine)
        raise e

    @trainer.on(Events.ITERATION_COMPLETED(every=args.sample_interval))
    def save_samples(engine):
        if args.dir_samples is not None:
            os.makedirs(args.dir_samples, exist_ok=True)
            samples = {
                k: v[:args.num_samples, ...].detach().cpu()
                for k, v in engine.state.output.items() if 'loss' not in k
            }
            for k, v in samples.items():
                if 'light' not in k:
                    tvutils.save_image(
                        v, os.path.join(args.dir_samples, k + '.png'))
            np.savetxt(os.path.join(args.dir_samples, 'light.txt'),
                       (samples['light_input'] -
                        samples['light_output']).numpy())

    # progress bar
    def attach_pbar(engine):
        pbar = ProgressBar()
        metric_names = [
            'face_loss', 'albedo_loss', 'normal_loss', 'light_loss'
        ]
        list(
            map(
                lambda name: RunningAverage(output_transform=lambda x: x[
                    name].item()).attach(engine, name), metric_names))
        pbar.attach(engine, metric_names=metric_names)

        @engine.on(Events.EPOCH_COMPLETED)
        def log_metrics(engine):
            pbar.log_message(
                f'Epoch {engine.state.epoch} done. Metrics: {engine.state.metrics}'
            )

    attach_pbar(trainer)
    attach_pbar(evaluator)

    trainer.run(train_dl, args.epochs)


if __name__ == '__main__':
    main(parse_arguments())