import torch
from .trainer import Trainer
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_module import make_model
import numpy as np
from modules.classifier import CosineClassifier
from torch.utils import data
import random
from .utils import get_batch, get_prototype
import math
import copy
from modules.deeplab import DeeplabV3
from functools import partial
from modules.custom_bn import InPlaceABR


class AMP(Trainer):
    """
    Reimplementation of https://arxiv.org/abs/1902.11123
    Starts from a standard step 0 that should be trained with FT classifier (use FT method)
    The incremental steps use the warm_up that updates the classification weights by averaging the pixels of the class.
    No training is performed (but you can try it just setting a number of iteration/epochs != 0)
    """
    EPOCHS = 5

    def initialize(self, opts):
        super(AMP, self).initialize(opts)
        self.amp_alpha = opts.amp_alpha

    def warm_up_(self, dataset, epochs=1):
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = len(self.task.order)
        sum_features = torch.zeros(classes, model.head_channels).to(self.device)
        count_features = torch.zeros(classes).to(self.device)

        for ep in range(AMP.EPOCHS):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    out_size = images.shape[-2:]
                    out = model(images.unsqueeze(0), use_classifier=False)
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False).squeeze_(0)
                    cl = labels.unique().cpu().numpy()  # get list of classes
                    for c in cl:
                        if c != 255:
                            feat = out[:, labels == c]  # F x P (pixels of class c)
                            sum_features[c] += feat.mean(dim=1)  # F
                            count_features[c] += 1
        # we have finished computing features, now collect and imprint!
        for c in range(classes):
            if count_features[c] != 0:
                features = sum_features[c] / count_features[c]
                features = features / features.norm()
                model.cls.imprint_weights_class(features=features, cl=c, alpha=self.amp_alpha)


class WeightImprinting(Trainer):
    """
    Simple WI made pixel-wise.
    Starts from a standard step 0 that should be trained with COS classifier (use COS method)
    The incremental steps use the warm_up that setups the classification weights by averaging the pixels of the class.
    No training is performed (but you can try it just setting a number of iteration/epochs != 0)
    """
    def __init__(self, task, device, logger, opts):
        super().__init__(task, device, logger, opts)
        self.masking = True
        self.normalize_weight = False
        self.compute_score = False

    def warm_up_(self, dataset, epochs=5):
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        sum_features = torch.zeros(classes[-1], model.head_channels).to(self.device)
        oi_acc = torch.zeros(classes[-1], classes[0] - 1).to(self.device)
        count = torch.zeros(classes[-1]).to(self.device)
        for ep in range(epochs):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)

                    out = model(images.unsqueeze(0), use_classifier=False)  # .squeeze_(0)
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False).squeeze_(0)
                    # labels = F.interpolate(labels.float().view(1, 1, labels.shape[0], labels.shape[1]),
                    #                        size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
                    cl = labels.unique().cpu().numpy()  # get list of classes
                    for c in new_classes:
                        if c in cl:
                            feat = out[:, labels == c]  # F x P (pixels of class c)

                            if self.compute_score:
                                oi = model.cls.cls[0](out.unsqueeze(0)).squeeze(0)
                                oi = oi[:, labels == c]  # get scores for pixels of c only -> oi has size C x P_c
                                oi = oi[1:].softmax(dim=0)  # make softmax on base classes / bkg -> C_b x P_c
                                oi_acc[c - old_classes] += oi.mean(dim=1)

                            feat = F.normalize(F.normalize(feat, dim=0).sum(dim=1), dim=0)
                            sum_features[c - old_classes] += feat
                            count[c - old_classes] += 1

        # we have finished computing features, now collect and imprint!
        assert torch.any(count != 0), "Error, a novel class has no pixels!"
        features = F.normalize(sum_features, dim=1)
        if self.normalize_weight:
            if self.compute_score:
                scores = oi_acc / count.view(-1, 1)
                for c in range(classes[-1]):
                    f = (model.cls.cls[0].weight[1:] * scores[c].view(classes[0] - 1, 1, 1, 1))
                    features[c] += f.sum(dim=0).view(model.head_channels)
            else:
                features += model.cls.cls[0].weight[1:].mean(dim=0).view(-1).detach()
        model.cls.imprint_weights_step(step=self.task.step, features=features)


class DynamicWI(Trainer):
    """
    Re implementation of https://openaccess.thecvf.com/content_cvpr_2018/papers/Gidaris_Dynamic_Few-Shot_Visual_CVPR_2018_paper.pdf
    We made it pixel-wise.
    You should first run the step 0 and perform the cool down, then you can start with incremental steps.
    The incremental steps use the warm_up that setups the classification weights.
    No training is performed (but you can try it just setting a number of iteration/epochs != 0)
    """
    BATCH_SIZE = 4
    EPISODE = 2
    LR = 0.1
    ITER = 10

    def __init__(self, task, device, logger, opts):
        super(DynamicWI, self).__init__(task, device, logger, opts)
        self.dim = self.n_channels

        self.weights = nn.Module()
        self.weight_a = nn.Parameter(F.normalize(torch.ones((self.n_channels, 1, 1), device=self.device), dim=0))
        self.weights.register_parameter("weight_a", self.weight_a)

        self.weight_b = nn.Parameter(F.normalize(torch.ones((self.n_channels, 1, 1), device=self.device), dim=0))
        self.weights.register_parameter("weight_b", self.weight_b)

        self.keys = nn.Parameter(
            torch.FloatTensor(self.task.get_n_classes()[0], self.n_channels).normal_(0., math.sqrt(2 / self.dim)))
        self.weights.register_parameter("keys", self.keys)

        self.att_weight = nn.Linear(self.dim, self.dim)
        self.att_weight.weight.data.copy_(torch.eye(self.dim, self.dim) + torch.randn(self.dim, self.dim) * 0.001)
        self.att_weight.bias.data.zero_()
        self.weights.add_module("att_weight", self.att_weight)

        self.weights.to(self.device)

        self.use_attention = False

        DynamicWI.LR = opts.dyn_lr
        DynamicWI.ITER = opts.dyn_iter

    def compute_weight(self, inp, cls_weight):
        # input is a D dimensional prototype
        if self.use_attention:
            sum_weight = torch.zeros(self.dim, 1, 1).to(self.device)
            count_weight = 0
            for x in inp:
                x = self.weights.att_weight(x)  # DxD x D = DxD
                x = x / x.norm()
                keys = self.weights.keys / self.weights.keys.norm(dim=1, keepdim=True)  # CxD
                x = (keys @ x).softmax(dim=0)  # C
                sum_weight += (x.view(-1, 1, 1, 1) * cls_weight).sum(dim=0)
                count_weight += 1
            att_weight = sum_weight / count_weight
            weight = self.weights.weight_a * inp.mean(dim=0).view(-1, 1, 1) + self.weights.weight_b * att_weight
        else:
            weight = self.weights.weight_a * inp.mean(dim=0).view(-1, 1, 1)
        return weight

    def warm_up_(self, dataset, epochs=1):
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        for c in new_classes:
            weight = None
            count = 0
            while weight is None and count < 10:
                ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
                wc = get_prototype(model, ds, c, self.device, interpolate_label=False, return_all=True)
                if wc is not None:
                    weight = self.compute_weight(wc, model.cls.cls[0].weight)
                count += 1

            if weight is not None:
                model.cls.imprint_weights_class(weight, c)
            else:
                raise Exception(f"Unable to imprint weight of class {c} after {count} trials.")

    def cool_down(self, dataset, epochs=1):
        if self.step == 0:
            # instance a new model without DDP!
            model = make_model(self.opts, CosineClassifier(self.task.get_n_classes(), channels=self.n_channels))
            scaler = model.cls.scaler
            state = {}
            for k, v in self.model.state_dict().items():
                state[k[7:]] = v
            model.load_state_dict(state, strict=True)
            model = model.to(self.device)
            model.fix_bn()
            model.eval()

            # instance optimizer, criterion and data
            classes = np.arange(0, self.task.get_n_classes()[0])
            classes = classes[1:]  # remove bkg ONLY from sampling
            params = [{"params": model.cls.cls.parameters()}, {"params": self.weights.parameters()}]
            optimizer = torch.optim.SGD(params, lr=DynamicWI.LR)
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

            dataloader = data.DataLoader(dataset, batch_size=min(DynamicWI.BATCH_SIZE, len(dataset)), shuffle=True,
                                         num_workers=4, drop_last=True)
            it = iter(dataloader)
            loss_tot = 0.
            # start train loop
            for i in range(DynamicWI.ITER):
                loss_step = 0.
                optimizer.zero_grad()

                for e in range(DynamicWI.EPISODE):
                    weight = torch.zeros_like(model.cls.cls[0].weight)
                    K = random.choice([1, 2, 5])
                    cls = random.choice(classes)  # sample N classes
                    for c in range(self.task.get_n_classes()[0]):
                        if c == cls:
                            ds = dataset.get_k_image_of_class(cl=cls, k=K)  # get K images of class c
                            wc = get_prototype(model, ds, cls, self.device, return_all=True)
                            if wc is None:
                                # print("WC is None!!")
                                weight[c] = F.normalize(model.cls.cls[0].weight[c], dim=0)
                            else:
                                class_weight = self.compute_weight(wc, model.cls.cls[0].weight)
                                weight[c] = F.normalize(class_weight, dim=0)
                        else:
                            weight[c] = F.normalize(model.cls.cls[0].weight[c], dim=0)

                    # get a batch of images from dataloader
                    it, batch = get_batch(it, dataloader)
                    ds = dataset.get_k_image_of_class(cl=cls, k=DynamicWI.BATCH_SIZE)  # get K images of class c
                    im_ds = [ds[i][0].unsqueeze(0) for i in range(len(ds))]
                    lbl_ds = [ds[i][1].unsqueeze(0) for i in range(len(ds))]
                    images = torch.cat([batch[0], *im_ds], dim=0).to(self.device, dtype=torch.float32)
                    labels = torch.cat([batch[1], *lbl_ds], dim=0).to(self.device, dtype=torch.long)

                    with torch.no_grad():
                        out = model(images, use_classifier=False)
                        out = F.normalize(out, dim=1)
                    out = scaler * F.conv2d(out, weight)
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
                    loss = criterion(out, labels)
                    loss.backward()
                    loss_step += loss.item()
                optimizer.step()

                self.logger.add_scalar("loss_cool_down", loss_step / DynamicWI.EPISODE, i + 1)
                loss_tot += loss_step / DynamicWI.EPISODE
                if (i % 10) == 0:
                    self.logger.info(f"Cool down loss at iter {i + 1}: {loss_tot / 10}")
                    loss_tot = 0

            state = {}
            if self.distributed:
                for k, v in model.state_dict().items():
                    state["module." + k] = v
            else:
                state = model.state_dict()
            self.model.load_state_dict(state)

    def load_state_dict(self, checkpoint, strict=True):
        super().load_state_dict(checkpoint, strict)
        if self.step > 0:
            self.weights.load_state_dict(checkpoint['weights'])
            self.weights.to(self.device)

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(),
                 "weights": self.weights.state_dict()}
        return state
