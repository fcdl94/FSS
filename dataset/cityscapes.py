import os
import torch.utils.data as data
import numpy as np
from torch import from_numpy
from PIL import Image
import pickle5 as pkl
from .dataset import FSSDataset

eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
map_classes = {
    7: "road",          # 1
    8: "sidewalk",      # 2
    9: "parking",
    10: "rail truck",
    11: "building",     # 3
    12: "wall",         # 4
    13: "fence",        # 5
    14: "guard_rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",         # 6
    18: "pole_group",
    19: "light",        # 7
    20: "sign",         # 8
    21: "vegetation",   # 9
    22: "terrain",      # 10
    23: "sky",          # 11
    24: "person",       # 12
    25: "rider",        # 13
    26: "car",          # 14
    27: "truck",        # 15
    28: "bus",          # 16
    29: "caravan",
    30: "trailer",
    31: "train",        # 17
    32: "motocycle",    # 18
    33: "bicycle"       # 19
}


class Cityscapes(data.Dataset):

    def __init__(self, root, train=True, transform=None, cl19=False, target_transform=None):

        root = os.path.expanduser(root)
        base_dir = "cityscapes"
        ds_root = os.path.join(root, base_dir)
        splits_dir = os.path.join(ds_root, 'split')

        if train:
            split_f = os.path.join(splits_dir, 'fine_train.txt')
        else:
            split_f = os.path.join(splits_dir, 'fine_val.txt')

        with open(os.path.join(split_f), "r") as f:
            files = [x[:-1].split(' ') for x in f.readlines()]

        if train:
            self.class_to_images_ = pkl.load(open(splits_dir+'/inverse_dict_train.pkl', 'rb'))
        else:
            self.class_to_images_ = None

        self.images = [(os.path.join(ds_root, x[0]), os.path.join(ds_root, x[1])) for x in files]

        self.transform = transform

        self.target_transform = target_transform
        if cl19 and target_transform is not None:
            classes = eval_classes
            mapping = np.zeros((256,), dtype=np.int64) + 255
            for i, cl in enumerate(classes):
                mapping[cl] = i
            self.target_transform = lambda x: from_numpy(mapping[x])

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


class CityscapesFSSDataset(FSSDataset):
    def make_dataset(self, root, train):
        full_voc = Cityscapes(root, train, transform=None)
        return full_voc
