#!/usr/bin/env python3
import json

from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataset_syn(Dataset):
    def __init__(self, data_quality='4', data_sampling_step='100', data_amount='200k', transform=None):
        """
        Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
        creates from test set.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.transform = transform
        self.data_quality = data_quality
        self.data_sampling_step = data_sampling_step
        self.data_amount = data_amount

        self.path_label = self.get()

    def get(self):
        path_label = []
        
        text_path = './syn_dataset/s'+str(self.data_quality)+'_st'+str(self.data_sampling_step)+'_'+str(self.data_amount)+'.txt'
        print(text_path)
        with open(text_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                path = line.strip()
                label = line.strip().split('/')[-1].split('_')[1]

                path_label.append((path, int(label)))

        return path_label

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class ImageNetDataset_local(Dataset):
    def __init__(self, transform=None):
        """
        Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
        creates from test set.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.transform = transform
        self.path_label = self.get()

    def get(self):
        path_label = []
        text_path = './true_dataset/imagenet_val.txt'
        with open(text_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                path = line.strip().split(' ')[0]
                label = line.strip().split(' ')[1]
                path_label.append((path, int(label)))

        return path_label

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
