import os.path as osp
import torch.utils.data as data
from .dataset import FSSDataset
import numpy as np
import pickle5 as pkl

from PIL import Image

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}
groups_list = ['person', 'animals', 'vehicles', 'indoor']
groups = {
    'person': [15],
    'animals': [3, 8, 10, 12, 13, 17],
    'vehicles': [1, 2, 4, 6, 7, 14, 19],
    'indoor': [5, 9, 11, 16, 18, 20]
}


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        train (bool): Use train (True) or test (False) split
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root="data", train=True, transform=None, coco_labels=False):

        if train:
            split = 'train_ids'
        else:
            split = 'test_ids'

        self.root = osp.expanduser(root)
        self.transform = transform

        voc_root = osp.join(self.root, "voc/dataset/")
        splits_dir = osp.join(self.root, "voc/split/")

        if not osp.isdir(voc_root):
            raise RuntimeError(f'Dataset not found in {voc_root}.' +
                               f' Download it with download_voc and then link it into {voc_root}.')

        if train:
            self.class_to_images_ = pkl.load(open(splits_dir + 'inverse_dict_train.pkl', 'rb'))
        else:
            self.class_to_images_ = None

        self.images = np.load(osp.join(splits_dir, split + '.npy'))
        if coco_labels:
            annotation_folder = "annotations_coco"
        else:
            annotation_folder = "annotations"
        self.images = [(osp.join(voc_root, "images", i + ".jpg"), osp.join(voc_root, annotation_folder, i + ".png"))
                       for i in self.images]

    @property
    def class_to_images(self):
        return self.class_to_images_

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VOCFSSDataset(FSSDataset):
    def make_dataset(self, root, train):
        full_voc = VOCSegmentation(root, train, transform=None)
        return full_voc
