import torch
from .method import Method
import torch.nn as nn
from .utils import get_scheduler
from .segmentation_module import make_model
from modules.classifier import SPNetClassifier
from apex import amp
from apex.parallel import DistributedDataParallel
from utils.loss import UnbiasedCrossEntropy, UnbiasedKnowledgeDistillationLoss, KnowledgeDistillationLoss


class SPNet(Method):
    def initialize(self, opts):

        cls = SPNetClassifier(opts, self.task.get_order())

        self.model = make_model(opts, cls)

        if opts.fix_bn:
            self.model.fix_bn()

        # xxx Set up optimizer
        params = []
        params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr * opts.lr_head})

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


class SPNet_LwF(Method):
    def __init__(self, task, device, logger, opts):
        super().__init__(task, device, logger, opts)

        classifier = SPNetClassifier(opts, self.task.get_order())
        self.model = make_model(opts, classifier)
        if task.step > 0:
            cl_old = SPNetClassifier(opts, self.task.get_old_labels())
            self.model_old = make_model(opts, cl_old)
            # put the old model into distributed memory and freeze it
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        if opts.fix_bn:
            self.model.fix_bn()
            if self.model_old is not None:
                self.model_old.fix_bn()

        # xxx Set up optimizer
        params = []
        params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr * 10.})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=False)
        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'mean'
        if self.task.step == 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)
            self.model, self.optimizer = amp.initialize(self.model.to(self.device), self.optimizer,
                                                        opt_level=opts.opt_level)
        else:
            self.criterion = self.get_criterion(task, reduction)
            self.regularizer = self.get_regularizer()
            [self.model, self.model_old], optimizer = amp.initialize([self.model.to(device), self.model_old.to(device)],
                                                                     self.optimizer, opt_level=opts.opt_level)
            self.model_old = DistributedDataParallel(self.model_old, delay_allreduce=True)
        # Put the model on GPU
        self.model = DistributedDataParallel(self.model, delay_allreduce=True)

    def get_criterion(self, task, reduction):
        return nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

    def get_regularizer(self):
        return KnowledgeDistillationLoss()

    def initialize(self, opts):
        pass

    def load_state_dict(self, checkpoint, strict=True):
        if self.model_old is not None and not strict:
            self.model_old.load_state_dict(checkpoint["model"], strict=True)  # we are loading the old model
        if not strict:
            del checkpoint["model"]['module.cls.class_emb']
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        if strict:  # if strict, we are in ckpt (not step) so load also optim and scheduler
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict()}
        return state


class SPNet_MiB(SPNet_LwF):
    def get_criterion(self, task, reduction):
        return UnbiasedCrossEntropy(ignore_index=255, reduction=reduction,
                                    old_cl=len(task.get_order()) - len(task.get_novel_labels()))

    def get_regularizer(self):
        return UnbiasedKnowledgeDistillationLoss()