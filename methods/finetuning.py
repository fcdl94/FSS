from .segmentation_module import make_model
from modules.classifier import IncrementalClassifier
import torch.nn as nn
from .utils import get_scheduler
from .method import Method
import torch
import torch.nn.functional as F


class FineTuning(Method):
    def initialize(self, opts):
        self.n_channels = 256
        self.model = make_model(opts, IncrementalClassifier(self.task.get_n_classes(), channels=self.n_channels))

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


class FineTuningClassifier(Method):
    def initialize(self, opts):
        self.n_channels = 256
        self.model = make_model(opts, IncrementalClassifier(self.task.get_n_classes(), channels=self.n_channels))

        if opts.fix_bn:
            self.model.fix_bn()

        # xxx Set up optimizer
        params = [{"params": filter(lambda p: p.requires_grad, self.model.cls[-1].parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr * opts.lr_cls}]

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=False)
        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'none'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)


class AMP(FineTuning):
    EPOCHS = 5

    def initialize(self, opts):
        super(AMP, self).initialize(opts)
        self.amp_alpha = opts.amp_alpha

    def warm_up(self, dataset, epochs=1):
        self.model.eval()
        classes = len(self.task.order)
        sum_features = torch.zeros(classes, self.model.module.head_channels).to(self.device)
        count_features = torch.zeros(classes).to(self.device)

        for ep in range(AMP.EPOCHS):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    out_size = images.shape[-2:]
                    out = self.model(images.unsqueeze(0), use_classifier=False)
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
                self.model.module.cls.imprint_weights_class(features=features, cl=c, alpha=self.amp_alpha)
