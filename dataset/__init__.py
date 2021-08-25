from .voc import VOCFSSDataset, VOCSegmentation
from .cityscapes import CityscapesFSSDataset
from .coco import COCOFSS, COCO, COCOStuffFSS
from .ade import AdeSegmentation
from .transform import Compose, RandomScale, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, \
    CenterCrop, Resize, RandomResizedCrop, ColorJitter, PadCenterCrop
import random
from .utils import Subset, MyImageFolder, RandomDataset

TRAIN_CV = 0.8


def get_dataset(opts, task, train=True):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'cts':
        dataset = CityscapesFSSDataset
        train_transform = transform.Compose([
            transform.RandomScale((0.7, 2)),  # Using RRC should be (0.25, 0.75)
            transform.RandomCrop(opts.crop_size),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transform.Compose([
            transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    elif opts.dataset == 'voc' or 'coco' in opts.dataset:
        if opts.dataset == 'voc':
            dataset = VOCFSSDataset
        else:
            if 'stuff' in opts.dataset:
                dataset = COCOStuffFSS
            else:
                dataset = COCOFSS

        train_transform = Compose([
            RandomScale((0.5, 2)),
            RandomCrop(opts.crop_size, pad_if_needed=True),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
        val_transform = Compose([
            # Resize(size=opts.crop_size_test),
            PadCenterCrop(size=opts.crop_size_test),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
        test_transform = Compose([
            # PadCenterCrop(size=opts.crop_size_test),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise NotImplementedError

    if train:
        if opts.cross_val:
            train_dst = dataset(root=opts.data_root, task=task, train=True, transform=None)
            train_len = int(TRAIN_CV * len(train_dst))
            idx = list(range(len(train_dst)))
            random.shuffle(idx)
            train_dst = Subset(train_dst, idx[:train_len], train_transform)
            val_dst = Subset(train_dst, idx[train_len:], val_transform)
            train_dst_noaug = Subset(train_dst, idx[:train_len], test_transform)
        else:
            train_dst = dataset(root=opts.data_root, task=task, train=True, transform=train_transform, masking=opts.masking)
            train_dst_noaug = dataset(root=opts.data_root, task=task, train=True, transform=test_transform, masking=opts.masking)
            val_dst = dataset(root=opts.data_root, task=task, train=False, transform=val_transform)

        return train_dst, val_dst, train_dst_noaug
    else:
        test_dst_all = dataset(root=opts.data_root, task=task, train=False, transform=test_transform)
        test_dst_novel = dataset(root=opts.data_root, task=task, train=False, transform=test_transform,
                                 masking=True)
        return test_dst_all, test_dst_novel
