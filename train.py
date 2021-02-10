# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Loss
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers import ProgressBar

import models
import datasets


def parse_arguments():
    parser = argparse.ArgumentParser('SfSNet Training')
    parser.add_argument('--epochs', type=int, default=5, action='store')
    parser.add_argument('--batch_size', type=int, default=8, action='store')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        action='store')
    parser.add_argument('--root_synthetic',
                        type=str,
                        default=None,
                        action='store')
    parser.add_argument('--root_celeba',
                        type=str,
                        default=None,
                        action='store')
    parser.add_argument('--checkpoints',
                        type=str,
                        default='checkpoints',
                        action='store')
    parser.add_argument('--pretrained', type=str, default=None, action='store')
    parser.add_argument('--image_size', type=int, default=128, action='store')
    parser.add_argument('--num_workers', type=int, default=2, action='store')
    parser.add_argument('--apply_mask', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    print('Parsed Arguments:', args)
    return args


def main(args):
    train_ds = datasets.SyntheticFace(
        args.root_synthetic,
        train=True,
        image_size=args.image_size,
    )
    valid_ds = datasets.SyntheticFace(
        args.root_synthetic,
        train=False,
        image_size=args.image_size,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
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

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.2,
        verbose=True,
    )

    def train_step(engine, batch):
        net.train()
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
        losses = {
            k + '_loss': F.mse_loss(output[k], target[k])
            for k in target
        }
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return losses | output

    trainer = Engine(train_step)
    timer = Timer(average=True)
    checkpoint = ModelCheckpoint(
        args.checkpoints,
        'sfsnet',
        n_saved=10,
        atomic=True,
        create_dir=True,
        save_as_state_dict=True,
    )

    metric_names = ['face_loss', 'albedo_loss', 'normal_loss', 'light_loss']

    list(
        map(
            lambda name: RunningAverage(output_transform=lambda x: x[
                name].item()).attach(trainer, name), metric_names))

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=metric_names)

    @trainer.on(Events.EPOCH_COMPLETED)
    def scheduler_step(engine):
        scheduler.step()

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=100),
        handler=checkpoint,
        to_save={'net': net},
    )

    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        pause=Events.EPOCH_COMPLETED,
        resume=Events.ITERATION_STARTED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_time(engine):
        pbar.log_message(
            f'Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]'
        )
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_metrics(engine):
        pbar.log_message(
            f'Epoch {engine.state.epoch} done. Metrics: {engine.state.metrics}'
        )

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt):
            engine.terminate()
            print('KeyboardInterrupt, saving')
            checkpoint(engine, to_save={'net': net})
        else:
            engine.terminate()
            checkpoint(engine, to_save={'net': net})
            raise e

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
            losses = {
                k + '_loss': F.mse_loss(output[k], target[k])
                for k in target
            }
            return losses | output

    evaluator = Engine(valid_step)

    metric_names = ['face_loss', 'albedo_loss', 'normal_loss', 'light_loss']

    list(
        map(
            lambda name: RunningAverage(output_transform=lambda x: x[
                name].item()).attach(evaluator, name), metric_names))

    pbar = ProgressBar()
    pbar.attach(evaluator, metric_names=metric_names)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run(valid_dl)

    trainer.run(train_dl, args.epochs)


if __name__ == '__main__':
    main(parse_arguments())