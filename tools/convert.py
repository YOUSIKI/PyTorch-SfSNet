# -*- encoding=utf-8 -*-

import os
import glob
import tqdm
import shutil
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser('Convert Original Dataset to SfSDataset')
    parser.add_argument('-m', '--mode', type=str)
    parser.add_argument('-s', '--src', type=str)
    parser.add_argument('-d', '--dst', type=str)
    args = parser.parse_args()
    return args


def convert_synthetic(src, dst):
    templates = list(sorted(glob.glob(os.path.join(src, '*', '*face*.png'))))
    templates = list(map(lambda s: s.replace('face', '%s'), templates))
    templates = list(map(lambda s: s.replace('png', '%s'), templates))
    table = {
        'face': 'png',
        'mask': 'png',
        'light': 'txt',
        'albedo': 'png',
        'normal': 'png',
    }
    for index, template in tqdm.tqdm(enumerate(templates),
                                     total=len(templates)):
        for k, v in table.items():
            srcfile = template % (k, v)
            dstfile = os.path.join(dst, '%s/%08d.%s' % (k, index, v))
            os.makedirs(os.path.dirname(dstfile), exist_ok=True)
            shutil.copy2(srcfile, dstfile)


def convert_celeba(src, dst):
    filenames = list(sorted(glob.glob(os.path.join(src, '*.png'))))
    for index, srcfile in tqdm.tqdm(enumerate(filenames),
                                    total=len(filenames)):
        dstfile = os.path.join(dst, 'face/%08d.png' % index)
        os.makedirs(os.path.dirname(dstfile), exist_ok=True)
        shutil.copy2(srcfile, dstfile)


if __name__ == '__main__':
    args = parse_arguments()
    if args.mode.lower().startswith('syn'):
        convert_synthetic(args.src, args.dst)
    else:
        convert_celeba(args.src, args.dst)
