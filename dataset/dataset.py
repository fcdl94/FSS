import torch.utils.data as data
from torch import from_numpy
import numpy as np
from .utils import Subset, ConcatDataset
import random


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
        self.class_to_images = {}

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        if train:
            self.inverted_order[255] = 255
            if task.use_bkg:
                self.inverted_order[0] = task.background_label
        else:
            if task.use_bkg:
                self.inverted_order[0] = task.background_label
            self.set_up_void_test()

        self.multi_idxs = False
        if not train:
            # in test we always use all images
            idxs = list(range(len(self.full_data)))
            target_transform = self.get_mapping_transform(self.labels, masking=masking, masking_value=masking_value)
            self.class_to_images = self.full_data.class_to_images

        elif step == 0 or task.nshot == -1:
            # if we use all images we are also sampling images not useful - such as images with only ignore pixels!
            # overlapped setup
            if task.disjoint:
                idxs = {x for x in range(len(self.full_data))}
                for cl, img_set in self.full_data.class_to_images.items():
                    if cl not in self.labels and (cl != 0 or not task.use_bkg):
                        idxs = idxs.difference(img_set)
                idxs = list(idxs)
                target_transform = self.get_mapping_transform(self.labels, masking, masking_value)
            else:
                idxs = {}
                for cl in self.labels:
                    idxs.update(self.full_data.class_to_images[cl])
                idxs = list(idxs)
                target_transform = self.get_mapping_transform(self.labels, masking, masking_value)
            for cl in self.labels:
                self.class_to_images[cl] = []
                for new_idx, idx in enumerate(idxs):
                    if idx in self.full_data.class_to_images[cl]:
                        self.class_to_images[cl].append(new_idx)
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
            count = 0
            if 0 in idxs:
                dts_list.append(Subset(self.full_data, idxs[0],
                                       transform=transform, target_transform=target_transform[0]))
                for cl in self.labels_old:
                    self.class_to_images[cl] = []
                    for new_idx, idx in enumerate(idxs):
                        if idx in self.full_data.class_to_images[cl]:
                            self.class_to_images[cl].append(new_idx)
                count += len(idxs[0])
            for i in range(1, len(self.labels)+1):
                dts_list.append(Subset(self.full_data, idxs[i],
                                       transform=transform, target_transform=target_transform[i]))
                self.class_to_images[self.labels[i-1]] = list(range(count, count+len(idxs[i])))
                count += len(idxs[i])

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

    def get_k_image_of_class(self, cl, k):
        assert cl < len(self.order) and cl != 0, f"Class must be in the actual task! Obtained {cl}"

        cl = self.order[cl]  # map to original mapping!
        assert len(self.class_to_images[cl]) >= k, f"There are no K images available for class {cl}."
        id_list = random.sample(self.class_to_images[cl], k=k)
        ret_images = []
        for i in id_list:
            ret_images.append(self[i])
        return ret_images


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return from_numpy(self.mapping[x])