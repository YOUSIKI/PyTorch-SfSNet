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

import data
import models


def parse_arguments():
    parser = argparse.ArgumentParser('SfSNet Training')
    # datasets
    parser.add_argument('--dataset_synthetic',
                        type=str,
                        default=None,
                        action='store',
                        help='path to Synthetic dataset')
    parser.add_argument('--dataset_celeba',
                        type=str,
                        default=None,
                        action='store',
                        help='path to CelebA dataset')
    # dataloader
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        action='store',
                        help='dataloader batch size (default: 16)')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        action='store',
                        help='dataloader num_workers (default: 4)')
    # epochs
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        action='store',
                        help='training epochs (default: 5)')
    # optimizer
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        action='store',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--scheduler_step_size',
                        type=int,
                        default=1,
                        action='store',
                        help='lr_scheduler step size (dedault 1)')
    parser.add_argument('--scheduler_gamma',
                        type=float,
                        default=0.2,
                        action='store',
                        help='lr_scheduler step size (dedault 0.2)')
    # checkpoints
    parser.add_argument('--checkpoints_dir',
                        type=str,
                        default='ckpts',
                        action='store',
                        help='directory to save checkpoints (default: ckpts)')
    parser.add_argument('--checkpoints_pre',
                        type=str,
                        default='sfsnet',
                        action='store',
                        help='checkpoints prefix (default: sfsnet)')
    parser.add_argument('--checkpoints_int',
                        type=int,
                        default=100,
                        action='store',
                        help='saving interval (iterations) (default: 100)')
    parser.add_argument('--checkpoints_num',
                        type=int,
                        default=5,
                        action='store',
                        help='num of kept checkpoints (default: 5)')
    # samples
    parser.add_argument('--samples_dir',
                        type=str,
                        default='samples',
                        action='store',
                        help='directory to save samples (default: samples)')
    parser.add_argument('--samples_num',
                        type=str,
                        default=4,
                        action='store',
                        help='number of samples to dump (default: 4)')
    parser.add_argument('--samples_int',
                        type=int,
                        default=200,
                        action='store',
                        help='saving samples interval (default: 200)')
    # load pretrained parameters
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        action='store',
                        help='load pretrained weight')
    # teacher model
    parser.add_argument('--teacher',
                        type=str,
                        default=None,
                        action='store',
                        help='load teacher model weight')
    # apply mask on images
    parser.add_argument('--apply_mask',
                        action='store_true',
                        help='apply mask on images (default: False)')
    # cuda support
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enable CUDA (default: False)')
    parser.add_argument('--cudnn_benchmark',
                        action='store_true',
                        help='Enable CUDNN benchmark (default: False)')
    args = parser.parse_args()
    print('Parsed Arguments:', args)
    return args


def main(args):
    # setup device
    if args.cuda and not torch.cuda.is_available():
        raise ValueError('CUDA support is not available, disabled')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # setup dataset
    train_ds = data.DatasetX()
    valid_ds = data.DatasetX()
    if args.dataset_synthetic:
        train_ds.append(
            data.FolderDataset(
                args.dataset_synthetic,
                split='train',
                device=device,
            ))
        valid_ds.append(
            data.FolderDataset(
                args.dataset_synthetic,
                split='valid',
                device=device,
            ))
    if args.dataset_celeba:
        teacher = models.SfSNet().to(device)
        teacher.load_state_dict(torch.load(args.teacher, map_location=device))
        train_ds.append(
            data.FolderDataset(
                args.dataset_celeba,
                split='train',
                device=device,
                teacher=teacher,
            ))
        valid_ds.append(
            data.FolderDataset(
                args.dataset_celeba,
                split='valid',
                device=device,
                teacher=teacher,
            ))

    # setup dataloader
    train_dl = data.DataLoaderX(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        # pin_memory=True,
    )
    valid_dl = data.DataLoaderX(
        dataset=valid_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        # pin_memory=True,
    )

    # setup model
    net = models.SfSNet().to(device)

    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained, map_location=device))

    # setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
        verbose=True,
    )

    # setup criterions
    criterion = {
        'face': F.l1_loss,
        'light': F.mse_loss,
        'albedo': F.l1_loss,
        'normal': F.l1_loss,
    }

    # universal engine step
    def universal_step(is_train):
        def step(engine, inputs):
            if is_train:
                net.train()
            else:
                net.eval()

            if is_train:
                grad_context = torch.enable_grad
            else:
                grad_context = torch.no_grad

            with grad_context():
                # move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # apply mask on images
                def mask(x):
                    if args.apply_mask:
                        return x * inputs['mask']
                    else:
                        return x

                outputs = net(inputs)

                # calculate losses
                losses = {
                    k: v(mask(outputs[k]), mask(inputs[k])) if k
                    in 'albedo normal'.split() else v(outputs[k], inputs[k])
                    for k, v in criterion.items()
                }

                # backward and optimization
                if is_train:
                    optimizer.zero_grad()
                    sum(losses.values()).backward()
                    optimizer.step()

                # generate fake results
                fakes = net(inputs, fake_forward=True)

                # tag dicts and return
                fakes = {f'{k}_fake': v for k, v in fakes.items()}
                losses = {f'{k}_loss': v for k, v in losses.items()}
                inputs = {f'{k}_input': v for k, v in inputs.items()}
                outputs = {f'{k}_output': v for k, v in outputs.items()}

            return inputs | outputs | losses | fakes

        return step

    # setup engine
    train_engine = Engine(universal_step(is_train=True))
    valid_engine = Engine(universal_step(is_train=False))

    # setup checkpoints
    checkpointer = ModelCheckpoint(
        dirname=args.checkpoints_dir,
        filename_prefix=args.checkpoints_pre,
        n_saved=args.checkpoints_num,
        atomic=True,
        require_empty=False,
        create_dir=True,
        save_as_state_dict=True,
    )

    @train_engine.on(Events.EPOCH_COMPLETED)
    def save_checkpoint_on_epoch_completed(engine):
        checkpointer(engine, {'net': net})

    @train_engine.on(Events.ITERATION_COMPLETED(every=args.checkpoints_int))
    def save_checkpoint_on_interation_completed(engine):
        checkpointer(engine, {'net': net})

    @train_engine.on(Events.EXCEPTION_RAISED)
    def save_checkpoint_on_excetion_raised(engine, exception):
        engine.terminate()
        checkpointer(engine, {'net': net})
        if not isinstance(exception, KeyboardInterrupt):
            raise exception

    # setup progress bar and metrics
    def attach_pbar_metrics(engine):
        metric_names = [
            'face_loss',
            'light_loss',
            'albedo_loss',
            'normal_loss',
        ]

        def create_running_average(name):
            metric = RunningAverage(output_transform=lambda d: d[name].item())
            metric.attach(engine, name)

        for name in metric_names:
            create_running_average(name)

        pbar = ProgressBar()

        pbar.attach(engine, metric_names=metric_names)

        @engine.on(Events.EPOCH_COMPLETED)
        def log_metrics(engine):
            pbar.log_message(
                f'Epoch {engine.state.epoch} metrics: {engine.state.metrics}')

    attach_pbar_metrics(train_engine)
    attach_pbar_metrics(valid_engine)

    # setup sampler
    def sampler(engine, dirname):
        os.makedirs(dirname, exist_ok=True)
        for k, v in engine.state.output.items():
            if 'loss' not in k:
                v = v[0:args.samples_num].detach().cpu()
                if 'light' not in k:
                    tvutils.save_image(
                        v,
                        os.path.join(dirname, f'{k}.png'),
                        normalize=False,
                    )
                else:
                    np.savetxt(os.path.join(dirname, f'{k}.txt'), v.numpy())

    @train_engine.on(Events.ITERATION_COMPLETED(every=args.samples_int))
    def save_samples_on_train_engine_iteration(engine):
        epoch = '%03d' % train_engine.state.epoch
        iteration = '%06d' % engine.state.iteration
        dirname = os.path.join(args.samples_dir, 'train', epoch, iteration)
        sampler(engine, dirname)

    @valid_engine.on(Events.ITERATION_COMPLETED(every=args.samples_int))
    def save_samples_on_valid_engine_iteration(engine):
        epoch = '%03d' % train_engine.state.epoch
        iteration = '%06d' % engine.state.iteration
        dirname = os.path.join(args.samples_dir, 'valid', epoch, iteration)
        sampler(engine, dirname)

    # run validation after each epoch
    @train_engine.on(Events.EPOCH_COMPLETED)
    def run_validation_after_each_epoch(engine):
        valid_engine.run(valid_dl)

    # adjust learning rate after each epoch
    @train_engine.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate_after_each_epoch(engine):
        scheduler.step()

    # release GPU cache
    if args.cuda:

        @train_engine.on(Events.ITERATION_COMPLETED)
        def release_gpu_cache_on_train_engine_iteration(engine):
            torch.cuda.empty_cache()

        @valid_engine.on(Events.ITERATION_COMPLETED)
        def release_gpu_cache_on_valid_engine_iteration(engine):
            torch.cuda.empty_cache()

    # start train_engine
    train_engine.run(train_dl, args.epochs)


if __name__ == '__main__':
    main(parse_arguments())