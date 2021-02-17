# -*- encoding=utf-8 -*-

import tqdm
import torch
import argparse
from models import SfSNet
from data import FolderDataset, DataLoaderX


def parse_arguments():
    parser = argparse.ArgumentParser(
        'Construct CelebA Dataset For Next Stage Using Pretrained Model')
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-w', '--weights', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    args = parser.parse_args()
    return args


def main(args):
    dataset = FolderDataset(args.dataset, split='all')
    dataloader = DataLoaderX(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             num_workers=4)
    net = SfSNet()
    net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    net.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm.tqdm(dataloader)):
            outputs = net(inputs)
            for name, item in outputs.items():
                if 'face' not in name:
                    for i in range(len(item)):
                        index = idx * args.batch_size + i
                        data = item[i].detach().cpu()
                        dataset[index] = {name: data}


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
