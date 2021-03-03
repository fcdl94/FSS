import torch.utils.data as data
from torch import from_numpy
import numpy as np
from .utils import Subset, ConcatDataset
import random
FILTER_FSL = False


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
        self.labels_fut = task.get_future_labels()
        assert not any(l in self.labels_old for l in self.labels), "Labels and labels_old must be disjoint sets"
        assert not any(l in self.labels_fut for l in self.labels), "Labels and labels_fut must be disjoint sets"

        self.masking_value = masking_value = 255
        self.class_to_images = {}

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        if train:
            self.inverted_order[255] = 255
            self.inverted_order[0] = 0
        else:
            self.inverted_order[0] = 0
            self.set_up_void_test()

        self.multi_idxs = False
        if not train:
            # in test we always use all images
            idxs = list(range(len(self.full_data)))
            # we mask unseen classes to 255. Usually masking is False and value=255, but can be set to True if you
            # want to evaluate only novel classes + bkg (old classes become 255)
            target_transform = self.get_mapping_transform(self.labels, masking=masking, masking_value=masking_value)
            self.class_to_images = self.full_data.class_to_images

        elif step == 0 or task.nshot == -1:
            # we filter images containing pixels of unseen classes (slow process but mandatory, I'm sorry).
            idxs = {x for x in range(len(self.full_data))}
            if task.disjoint:
                for cl, img_set in self.full_data.class_to_images.items():
                    if cl not in self.labels and (cl != 0):
                        idxs = idxs.difference(img_set)
            idxs = list(idxs)
            # this is useful to reorder the labels (not to mask since we already excluded the to-mask classes)
            target_transform = self.get_mapping_transform(self.labels, masking, masking_value)
            # this is helpful in case we need to sample images of some class
            index_map = {idx: new_idx for new_idx, idx in enumerate(idxs)}
            for cl in self.labels:
                self.class_to_images[cl] = []
                for idx in self.full_data.class_to_images[cl]:
                    if idx in index_map:
                        self.class_to_images[cl].append(index_map[idx])

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
                target_transform[0] = self.get_mapping_transform(self.labels_old, masking=True,
                                                                 masking_value=masking_value)
            for i, cl in enumerate(self.labels):
                images_of_class = self.full_data.class_to_images[cl]
                if FILTER_FSL:
                    # filter images containing unseen or actual classes
                    for cl_, img_set in self.full_data.class_to_images.items():
                        if cl_ != cl and cl_ not in self.labels_old and (cl_ != 0):
                            images_of_class = images_of_class.difference(img_set)
                idxs[i+1] = images_of_class[ishot*20: ishot*20+nshot]
                lbls = [cl] if masking else self.labels_old + [cl]
                # this is useful to reorder the labels (not to mask since we already excluded the to-mask classes)
                target_transform[i+1] = self.get_mapping_transform(lbls, masking=True, masking_value=masking_value)

        # make the subset of the dataset
        self.indices = []
        if not self.multi_idxs:  # step 0 or test
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
        # setup the mapping:
        # if masking=True, old classes become masking_value except the bkg. Ground Truth is remapped to the order.
        # if masking=False, all seen classes are reordered according the order. No seen class is excluded.
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