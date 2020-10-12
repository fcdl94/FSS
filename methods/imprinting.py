import torch
from .method import Method
import torch.nn as nn
from .utils import get_scheduler
import torch.nn.functional as F
from .segmentation_module import make_model
import numpy as np
from modules.classifier import CosineClassifier
from torch.utils import data
import random
from apex import amp


class CosineFT(Method):
    def initialize(self, opts):

        self.n_channels = 256
        self.model = make_model(opts, CosineClassifier(self.task.get_n_classes(), channels=self.n_channels))

        if opts.fix_bn:
            self.model.fix_bn()

        # xxx Set up optimizer
        params = []
        params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr * opts.lr_head})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr * opts.lr_cls})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=False)

        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'none'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)


class CosineFTC(Method):
    def initialize(self, opts):
        self.n_channels = 256
        self.model = make_model(opts, CosineClassifier(self.task.get_n_classes(), channels=self.n_channels))

        if opts.fix_bn:
            self.model.fix_bn()

        # xxx Set up optimizer
        params = []
        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls[-1].parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr * opts.lr_cls})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=False)

        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'none'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)


class WeightImprinting(CosineFT):
    def warm_up(self, dataset, epochs=5):
        self.model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes+classes[-1])
        sum_features = torch.zeros(classes[-1], self.model.module.head_channels).to(self.device)
        count_features = torch.zeros(classes[-1]).to(self.device)
        for ep in range(epochs):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    out = self.model(images.unsqueeze(0), use_classifier=False)  # .squeeze_(0)
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False).squeeze_(0)
                    # labels = F.interpolate(labels.float().view(1, 1, labels.shape[0], labels.shape[1]),
                    #                        size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
                    cl = labels.unique().cpu().numpy()  # get list of classes
                    for c in new_classes:
                        if c in cl:
                            feat = out[:, labels == c]  # F x P (pixels of class c)
                            sum_features[c-old_classes] += F.normalize(F.normalize(feat, dim=0).sum(dim=1), dim=0)
                            count_features[c-old_classes] += 1
        # we have finished computing features, now collect and imprint!
        assert torch.any(count_features != 0), "Error, a novel class has no pixels!"
        features = F.normalize(sum_features, dim=1)
        self.model.module.cls.imprint_weights_step(step=self.task.step, features=features)


class Prototypes(CosineFT):
    def get_prototype(self, out, labels, cl):
        # out must be C x H x W, labels must be H x W, cl is integer
        f = out[:, labels == cl]
        return F.normalize(F.normalize(f, dim=0).sum(dim=1), dim=0)

    def cool_down(self, dataset, epochs=1):
        self.model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        sum_features = torch.zeros(classes[-1], self.model.module.head_channels).to(self.device)
        count_features = torch.zeros(classes[-1]).to(self.device)
        for ep in range(epochs):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long).flatten()
                    out = self.model(images.unsqueeze(0), use_classifier=False)  # .squeeze_(0)
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False).squeeze_(0)
                    out = out.view(out.shape[0], -1)  # flatten
                    # labels = F.interpolate(labels.float().view(1, 1, labels.shape[0], labels.shape[1]),
                    #                        size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
                    cl = labels.unique().cpu().numpy()  # get list of classes
                    for c in new_classes:
                        if c in cl:
                            sum_features[c - old_classes] += self.get_prototype(out, labels, c)
                            count_features[c - old_classes] += 1
        # we have finished computing features, now collect and imprint!
        assert torch.any(count_features != 0), "Error, a class has no pixels!"
        features = F.normalize(sum_features, dim=1)
        self.model.module.cls.imprint_weights_step(step=self.task.step, features=features)


class DynamicWI(CosineFT):
    LR = 0.01
    ITER = 100
    BATCH_SIZE = 20

    def __init__(self, task, device, logger, opts):
        super(DynamicWI, self).__init__(task, device, logger, opts)
        self.weight = nn.Parameter(torch.ones(self.n_channels, device=self.device))

    def get_prototype(self, model, ds, cl, interpolate_label=True):
        protos = []
        with torch.no_grad():
            for img, lbl in ds:
                img, lbl = img.to(self.device), lbl.to(self.device)
                out = model(img.unsqueeze(0), use_classifier=False).detach()
                if interpolate_label:  # to match output size
                    lbl = F.interpolate(lbl.float().view(1, 1, lbl.shape[0], lbl.shape[1]),
                                        size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
                else:  # interpolate output to match label size
                    out = F.interpolate(out, size=img.shape[-2:], mode="bilinear", align_corners=False)
                out = out.squeeze(0)
                out = out.view(out.shape[0], -1).t()  # (HxW) x F
                lbl = lbl.flatten()  # Now it is (HxW)
                if (lbl == cl).float().sum() > 0:
                    protos.append(self.norm_mean(out[lbl == cl, :]))
        protos = torch.cat(protos, dim=0)
        return protos.mean(dim=0)

    @staticmethod
    def norm_mean(x):
        # x should be N x F, return 1 x F
        return F.normalize(x, dim=1).mean(dim=0, keepdim=True)

    @staticmethod
    def get_batch(it, dataloader):
        try:
            batch = next(it)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            it = iter(dataloader)
            batch = next(it)
        return it, batch

    def cool_down(self, dataset, epochs=1):
        model = self.model.module  # we should get the one without DDP
        with self.model.no_sync():
            if self.step == 0:
                classes = self.task.get_novel_labels()
                params = [{"params": filter(lambda p: p.requires_grad, model.cls.parameters())}, {"params": self.weight}]
                optimizer = torch.optim.SGD(params, lr=DynamicWI.LR, momentum=0.9, nesterov=False)
                # [model, self.weight], optimizer = amp.initialize([model.to(self.device), self.weight.to(self.device)],
                #                                                 optimizer, opt_level=self.opts.opt_level)
                criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

                dataloader = data.DataLoader(dataset, batch_size=min(DynamicWI.BATCH_SIZE, len(dataset)), shuffle=True,
                                             num_workers=4, drop_last=True)
                it = iter(dataloader)

                classifier_weights = model.cls.cls[0].weight.data.clone()

                for i in range(DynamicWI.ITER):
                    model.cls.cls[0].weight.data = classifier_weights
                    optimizer.zero_grad()
                    N = random.randint(1, 5)
                    K = random.choice([1, 5, 10])
                    cls = random.choices(population=classes, k=N)  # sample N classes
                    for c in cls:
                        ds = dataset.get_k_image_of_class(cl=c, k=K)  # get K images of class c
                        wc = self.get_prototype(model, ds, c)
                        model.cls.cls[0].weight.data[c] = (self.weight * wc).view(1, self.n_channels, 1, 1)

                    # get a batch of images from dataloader
                    it, batch = self.get_batch(it, dataloader)
                    images = batch[0].to(self.device, dtype=torch.float32)
                    labels = batch[1].to(self.device, dtype=torch.long)

                    out = model(images)
                    loss = criterion(out, labels)
                    loss.backward()
                    #with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #    scaled_loss.backward()
                    optimizer.step()

                    if (i % 50) == 0:
                        self.logger.info(f"Cool down loss at iter {i}: {loss.item()}")

    def load_state_dict(self, checkpoint, strict=True):
        super().load_state_dict(checkpoint, strict)
        if self.step > 0:
            self.weight.data = checkpoint['weight']

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "weight": self.weight.data}
        return state
