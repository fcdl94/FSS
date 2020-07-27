import torch
from .method import FineTuning
import torch.nn as nn
from .utils import get_scheduler
import torch.nn.functional as F
from .segmentation_module import make_model
import numpy as np


class Classifier(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.cls = nn.ModuleList(
            [nn.Conv2d(channels, c, 1, bias=False) for c in classes])
        self.scaler = 10.

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = []
        for i, mod in enumerate(self.cls):
            out.append(self.scaler * F.conv2d(x, F.normalize(mod.weight, dim=1, p=2)))
        return torch.cat(out, dim=1)

    def imprint_weights(self, step, features):
        self.cls[step].weight.data = features.view_as(self.cls[step].weight.data)


class CosineFT(FineTuning):
    def initialize(self, opts):

        head_channels = 256
        self.model = make_model(opts, head_channels, Classifier(head_channels, self.task.get_n_classes()))

        if opts.fix_bn:
            self.model.fix_bn()

        # xxx Set up optimizer
        params = []
        params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*10.})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*10.})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=False)

        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'mean'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)


class WeightImprinting(CosineFT):
    EPOCHS = 1

    def warm_up(self, dataset):
        self.model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes+classes[-1])
        sum_features = torch.zeros(classes[-1], self.model.module.head_channels).to(self.device)
        count_features = torch.zeros(len(self.task.get_novel_labels())).to(self.device)
        for ep in range(WeightImprinting.EPOCHS):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    out = self.model(images.unsqueeze(0), use_classifier=False).squeeze_(0)
                    labels = F.interpolate(labels.float().view(1, 1, labels.shape[0], labels.shape[1]),
                                           size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
                    cl = labels.unique().cpu().numpy()  # get list of classes
                    for c in new_classes:
                        if c in cl:
                            feat = out[:, labels == c]  # F x P (pixels of class c)
                            sum_features[c-old_classes] += F.normalize(F.normalize(feat, dim=0).mean(dim=1), dim=0)
                            count_features[c-old_classes] += 1
        # we have finished computing features, now collect and imprint!
        assert torch.any(count_features != 0), "Error, a novel class has no pixels!"
        features = sum_features / count_features.view(-1, 1)
        features = F.normalize(features, dim=0)
        self.model.module.cls.imprint_weights(self.task.step, features)


class AMP(FineTuning):
    EPOCHS = 1

    def warm_up(self, dataset):
        self.model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes+classes[-1])
        sum_features = torch.zeros(classes[-1], self.model.module.head_channels).to(self.device)
        count_features = torch.zeros(len(self.task.get_novel_labels())).to(self.device)
        for ep in range(WeightImprinting.EPOCHS):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    out = self.model(images.unsqueeze(0), use_classifier=False).squeeze_(0)
                    labels = F.interpolate(labels.float().view(1, 1, labels.shape[0], labels.shape[1]),
                                           size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
                    cl = labels.unique().cpu().numpy()  # get list of classes
                    for c in new_classes:
                        if c in cl:
                            feat = out[:, labels == c]  # F x P (pixels of class c)
                            sum_features[c-old_classes] += F.normalize(F.normalize(feat, dim=0).mean(dim=1), dim=0)
                            count_features[c-old_classes] += 1
        # we have finished computing features, now collect and imprint!
        assert torch.any(count_features != 0), "Error, a novel class has no pixels!"
        features = sum_features / count_features.view(-1, 1)
        features = F.normalize(features, dim=0)
        self.model.module.cls.imprint_weights(self.task.step, features)
