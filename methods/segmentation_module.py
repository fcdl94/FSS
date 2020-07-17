import torch
import torchvision
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional

import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial, reduce

import models
from modules import DeeplabV3


def make_model(opts, head_channels=256, cls=None):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex

    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory

    head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                     out_stride=opts.output_stride, pooling_size=opts.pooling)

    if cls is None:
        cls = nn.Conv2d(head_channels, opts.num_classes, 1)

    model = SegmentationModule(body, head, head_channels, cls)

    return model


class SegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classifier):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.head_channels = head_channels
        self.cls = classifier

    def _network(self, x):

        x_b = self.body(x)
        if isinstance(x_b, dict):
            x_b = x_b["out"]
        x_o = self.head(x_b)

        return x_o

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def forward(self, x):

        out = self._network(x)

        out_size = x.shape[-2:]
        sem_logits = self.cls(out)
        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

        return sem_logits

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
