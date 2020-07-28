from .segmentation_module import make_model
from modules.classifier import IncrementalClassifier
import torch.nn as nn
from .utils import get_scheduler
from .method import Method
import torch


class FineTuning(Method):
    def initialize(self, opts):
        head_channels = 256

        self.model = make_model(opts, head_channels, IncrementalClassifier(head_channels, self.task.get_n_classes()))

        if opts.fix_bn:
            self.model.fix_bn()

        # xxx Set up optimizer
        params = []
        params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr * 10.})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr * 10.})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=False)
        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'mean'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)


class FineTuningClassifier(Method):
    def initialize(self, opts):

        head_channels = 256

        class Classifier(nn.Module):
            def __init__(self, channels, classes):
                super().__init__()
                self.cls = nn.ModuleList(
                    [nn.Conv2d(channels, c, 1) for c in classes])

            def forward(self, x):
                out = []
                for mod in self.cls:
                    out.append(mod(x))
                return torch.cat(out, dim=1)

            def imprint_weights(self, step, features):
                self.cls[step].weight.data = features.view_as(self.cls[step].weight.data)

        self.model = make_model(opts, head_channels, Classifier(head_channels, self.task.get_n_classes()))

        if opts.fix_bn:
            self.model.fix_bn()

        # xxx Set up optimizer
        params = [{"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                   'weight_decay': opts.weight_decay, 'lr': opts.lr * 10.}]

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=False)
        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'mean'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)