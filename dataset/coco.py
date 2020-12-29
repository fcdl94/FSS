import os.path as osp
import torch.utils.data as data
import numpy as np
from torch import from_numpy
from PIL import Image
import pickle5 as pkl
from .dataset import FSSDataset

ignore_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]  # starting from 1=person


class COCO(data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, stuff=False):

        root = osp.expanduser(root)
        base_dir = "coco" if not stuff else "coco-stuff"
        ds_root = osp.join(root, base_dir)
        splits_dir = osp.join(ds_root, 'split')

        if train:
            split_f = osp.join(splits_dir, 'train.txt')
            folder = 'train2017'
        else:
            split_f = osp.join(splits_dir, 'val.txt')
            folder = 'val2017'

        ann_folder = "annotations"

        with open(osp.join(split_f), "r") as f:
            files = f.readlines()

        if train:
            path = '/inverse_dict_train_coco.pkl'
            self.class_to_images_ = pkl.load(open(splits_dir + path, 'rb'))
        else:
            path = '/inverse_dict_test_coco.pkl'
            self.class_to_images_ = pkl.load(open(splits_dir + path, 'rb'))

        self.images = [(osp.join(ds_root, "images", folder, x[:-1] + ".jpg"),
                        osp.join(ds_root, ann_folder, folder, x[:-1] + ".png")) for x in files]

        self.transform = transform
        self.target_transform = target_transform

    @property
    def class_to_images(self):
        return self.class_to_images_

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the label of segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])

        if self.transform is not None:
            img, target = self.transform(img, target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


class COCOFSS(FSSDataset):
    def make_dataset(self, root, train):
        data = COCO(root, train, transform=None)
        return data


class COCOStuffFSS(FSSDataset):
    def make_dataset(self, root, train):
        data = COCO(root, train, transform=None, stuff=True)
        return data
