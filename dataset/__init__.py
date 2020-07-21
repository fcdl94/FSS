from .voc import VOCFSSDataset
from .transform import Compose, RandomScale, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, CenterCrop
import torch


def get_dataset(opts, task):
    """ Dataset And Augmentation
    """
    train_transform = Compose([
        RandomScale((0.5, 1.5)),
        RandomCrop(opts.crop_size, pad_if_needed=True),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        CenterCrop(size=opts.crop_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    test_transform = Compose([
        CenterCrop(size=opts.crop_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    if opts.dataset == 'voc':
        dataset = VOCFSSDataset
    else:
        raise NotImplementedError

    train_dst = dataset(root=opts.data_root, task=task, train=True, transform=train_transform)

    if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst)-train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(root=opts.data_root, task=task, train=False, transform=val_transform)

    test_dst = dataset(root=opts.data_root, task=task, train=False, transform=test_transform)

    return train_dst, val_dst, test_dst
