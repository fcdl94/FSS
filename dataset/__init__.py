from .voc import VOCFSSDataset
from .transform import Compose, RandomScale, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, CenterCrop, Resize
import random
from .utils import Subset
TRAIN_CV = 0.8


def get_dataset(opts, task, train=True):
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
        Resize(size=opts.crop_size_test),
        CenterCrop(size=opts.crop_size_test),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    if opts.dataset == 'voc':
        dataset = VOCFSSDataset
    else:
        raise NotImplementedError

    if train:
        if opts.cross_val:
            train_dst = dataset(root=opts.data_root, task=task, train=True, transform=None,
                                masking=not opts.no_mask, masking_value=opts.masking)
            train_len = int(TRAIN_CV * len(train_dst))
            idx = list(range(len(train_dst)))
            random.shuffle(idx)
            train_dst = Subset(train_dst, idx[:train_len], train_transform)
            val_dst = Subset(train_dst, idx[train_len:], val_transform)
        else:
            train_dst = dataset(root=opts.data_root, task=task, train=True, transform=train_transform,
                                masking=not opts.no_mask, masking_value=opts.masking)
            val_dst = dataset(root=opts.data_root, task=task, train=False, transform=val_transform,
                              masking=not opts.no_mask, masking_value=opts.masking)
        return train_dst, val_dst
    else:
        test_dst_all = dataset(root=opts.data_root, task=task, train=False, transform=test_transform,
                               masking=False, masking_value=255)
        test_dst_novel = dataset(root=opts.data_root, task=task, train=False, transform=test_transform,
                                 masking=True, masking_value=255)
        return test_dst_all, test_dst_novel
