import torch
from .method import Method, FineTuning
import torch.nn as nn
from .utils import get_scheduler
from apex.parallel import DistributedDataParallel
from apex import amp
import pickle
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
from .segmentation_module import make_model


class SPNetClassifier(nn.Module):
    def __init__(self, opts, classes):
        super().__init__()
        datadir = f"data/{opts.dataset}"
        if opts.embedding == 'word2vec':
            class_emb = pickle.load(open(datadir + '/word_vectors/word2vec.pkl', "rb"))
        elif opts.embedding == 'fasttext':
            class_emb = pickle.load(open(datadir + '/word_vectors/fasttext.pkl', "rb"))
        elif opts.embedding == 'fastnvec':
            class_emb = np.concatenate([pickle.load(open(datadir + '/word_vectors/fasttext.pkl', "rb")),
                                        pickle.load(open(datadir + '/word_vectors/word2vec.pkl', "rb"))], axis=1)
        else:
            raise NotImplementedError(f"Embeddings type {opts.embeddings} is not known")

        self.class_emb = class_emb[classes]
        self.class_emb = F.normalize(torch.tensor(self.class_emb), p=2, dim=1)
        self.class_emb = torch.transpose(self.class_emb, 1, 0).float()
        self.class_emb = Parameter(self.class_emb, False)
        self.in_channels = self.class_emb.shape[0]

    def forward(self, x):
        return torch.matmul(x.permute(0, 2, 3, 1), self.class_emb).permute(0, 3, 1, 2)


class SPNet(FineTuning):
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
