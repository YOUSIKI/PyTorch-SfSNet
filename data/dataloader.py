# -*- encoding=utf-8 -*-

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

__all__ = ['DataLoaderX']


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
