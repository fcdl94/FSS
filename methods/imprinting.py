import torch
from .method import FineTuning
import torch.nn as nn
from .utils import get_scheduler
import torch.nn.functional as F
from .segmentation_module import make_model


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
