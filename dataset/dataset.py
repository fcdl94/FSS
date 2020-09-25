import torch.utils.data as data
from torch import from_numpy
import numpy as np
from .utils import Subset, ConcatDataset


class FSSDataset(data.Dataset):
    def __init__(self,
                 root,
                 task,
                 train=True,
                 transform=None,
                 masking=False):

        self.full_data = self.make_dataset(root, train)
        self.transform = transform

        step = self.step = task.step
        self.order = task.order
        self.labels = task.get_novel_labels()
        self.labels_old = task.get_old_labels(bkg=False)
        assert not any(l in self.labels_old for l in self.labels), "Labels and labels_old must be disjoint sets"
        self.masking_value = masking_value = 255

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        if train:
            self.inverted_order[255] = 255
            self.inverted_order[0] = task.background_label
        else:
            self.inverted_order[0] = task.background_label
            self.set_up_void_test()

        self.multi_idxs = False
        if not train:
            # in test we always use all images
            idxs = list(range(len(self.full_data)))
            target_transform = self.get_mapping_transform(self.labels, masking=masking, masking_value=masking_value)

        elif step == 0 or task.nshot == -1:
            # if we use all images we are also sampling images not useful - such as images with only ignore pixels!
            # overlapped setup
            idxs = {x for x in range(len(self.full_data))}
            if task.disjoint:
                for cl, img_set in self.full_data.class_to_images.items():
                    if cl not in self.labels and cl != 0:
                        idxs = idxs.difference(img_set)
                idxs = list(idxs)
                target_transform = self.get_mapping_transform(self.labels, masking, masking_value)
            else:
                for cl in self.labels:
                    idxs.update(self.full_data.class_to_images[cl])
                idxs = list(idxs)
                target_transform = self.get_mapping_transform(self.labels, masking, masking_value)
        else:  # Few Shot Learning
            self.multi_idxs = True
            idxs = {}
            target_transform = {}
            ishot = task.ishot
            nshot = task.nshot
            if task.input_mix == 'both':
                idxs[0] = []
                for cl in self.labels_old:
                    # 20 is max of nshot - taken from SPNet code
                    idxs[0].extend(self.full_data.class_to_images[cl][ishot*20: ishot*20+nshot])
                target_transform[0] = self.get_mapping_transform(self.labels_old, True, masking_value)
            for i, cl in enumerate(self.labels):
                idxs[i+1] = self.full_data.class_to_images[cl][ishot*20: ishot*20+nshot]
                lbls = [cl] if masking else self.labels_old + [cl]
                target_transform[i+1] = self.get_mapping_transform(lbls, True, masking_value)

        # make the subset of the dataset
        self.indices = []
        if not self.multi_idxs:
            self.dataset = Subset(self.full_data, idxs, transform=transform, target_transform=target_transform)
        else:
            dts_list = []
            if 0 in idxs:
                dts_list.append(Subset(self.full_data, idxs[0],
                                       transform=transform, target_transform=target_transform[0]))
            for i in range(1, len(self.labels)+1):
                dts_list.append(Subset(self.full_data, idxs[i],
                                       transform=transform, target_transform=target_transform[i]))
            self.dataset = ConcatDataset(dts_list)

    def get_mapping_transform(self, labels, masking, masking_value):
        # setup the mapping: old classes are masking_value and 0 and 255 have their value. Classes are reordered.
        if masking:
            tmp_labels = labels + [255, 0]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order
        mapping = np.full((256,), masking_value, dtype=np.uint8)
        for k in mapping_dict.keys():
            mapping[k] = mapping_dict[k]
        target_transform = LabelTransform(mapping)
        return target_transform

    def set_up_void_test(self):
        self.inverted_order[255] = 255

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        img, lbl = self.dataset[index]
        return img, lbl

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, train):
        raise NotImplementedError


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return from_numpy(self.mapping[x])