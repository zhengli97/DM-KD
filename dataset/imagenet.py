"""
get data loaders
"""
from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms
import imgaug.augmenters as iaa
from dataset.dataset_syn import ImageNetDataset_syn, ImageNetDataset_local

imagenet_list = ['imagenet', 'imagenette']


def get_syn_train_imagenet_dataloader(dataset='imagenet', batch_size=128, num_workers=16,
                            syn_data_amount='200k', syn_data_quality='4', syn_s_step='200', multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = ImageNetDataset_syn(data_quality=syn_data_quality, data_sampling_step=syn_s_step, data_amount=syn_data_amount, transform=train_transform)
    test_set = ImageNetDataset_local(transform=test_transform)

    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    return train_loader, train_sampler, test_loader
