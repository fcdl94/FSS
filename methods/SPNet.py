import torch
from .method import Method
import torch.nn as nn
from .utils import get_scheduler
from .segmentation_module import make_model
from modules.classifier import SPNetClassifier


class SPNet(Method):
    def initialize(self, opts):

        cls = SPNetClassifier(opts, self.task.get_order())

        self.model = make_model(opts, cls.in_channels, cls)

        if opts.fix_bn:
            self.model.fix_bn()

        # xxx Set up optimizer
        params = []
        params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*10.})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=False)
        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'mean'
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

    def load_state_dict(self, checkpoint, strict=True):
        if not strict:
            del checkpoint["model"]['module.cls.class_emb']
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        if strict:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
