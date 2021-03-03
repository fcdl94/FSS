import utils
import argparser
import os

import numpy as np
import random
import torch
from torch.utils import data

from dataset import get_dataset
from metrics import StreamSegMetrics
from task import Task

from modules.classifier import CosineClassifier
from modules.generators import GlobalGenerator, FeatGenerator
from methods.segmentation_module import make_model
from functools import partial
import torch.nn as nn
from torch.nn import functional as F
from inplace_abn import InPlaceABN
from modules import DeeplabV3


def get_batch(it, dataloader):
    try:
        batch = next(it)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        it = iter(dataloader)
        batch = next(it)
    return it, batch


def get_step_ckpt(opts, task_name):
    # Get step checkpoint
    if opts.step_ckpt is not None:
        path = opts.step_ckpt
    else:
        path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}.pth"

    # generate model from path
    if os.path.exists(path):
        step_checkpoint = torch.load(path, map_location="cpu")
        step_checkpoint['path'] = path
        return step_checkpoint
    else:
        raise FileNotFoundError(f"Step checkpoint not found in {path}")


# def generate_synth_feat(model, generator, images=None, labels=None):
#     real_feat, real_lbl = get_real_features(model, images, labels)
#     masked_feat, masked_lbl, real_feat, real_lbl = mask_features(real_feat, real_lbl)
#
#     gen_feat = generator(masked_feat).detach()
#
#     return gen_feat, masked_lbl


def get_real_features(model, images, labels):
    with torch.no_grad():
        _, _, feat = model(images, return_feat=True, return_body=True)
    # Downsample labels to match feat size (32x32)
    labels = F.interpolate(labels.float().unsqueeze(1), size=feat.shape[-2:], mode="nearest").long()

    return feat, labels.squeeze(1)  # get back to B x H x W


def get_bkg_proto(feat, labels):
    protos = []
    for i in range(feat.shape[0]):  # each image independently
        mask = labels[i] == 0
        if mask.sum() > 0:
            f = feat[i][:, mask].mean(dim=1)  # should  be 1xD
            protos.append(f.view(1, -1))
    return torch.cat(protos, dim=0)


def mask_features(feat, lbl):
    mask_feat = []
    real_feat = []
    mask_lbl = []
    real_lbl = []
    for i in range(feat.shape[0]):  # each image independently
        cls = lbl[i].unique()
        cls = cls[cls != 0]  # filter out bkg and void
        cls = cls[cls != 255]
        if len(cls) > 0:
            p = torch.ones_like(cls, dtype=torch.float32)  # make it float
            idx = p.multinomial(num_samples=1)
            cl = cls[idx]
            m = torch.eq(lbl[i], cl)
            mask_feat.append((feat[i] * m.float()).unsqueeze(0))
            real_feat.append(feat[i].unsqueeze(0))
            mask_lbl.append((lbl[i] * m.long()).unsqueeze(0))
            real_lbl.append((lbl[i]).unsqueeze(0))
    return torch.cat(mask_feat, dim=0), torch.cat(mask_lbl, dim=0).long(), \
           torch.cat(real_feat, dim=0), torch.cat(real_lbl, dim=0).long()


def train(dataloader, model, classifier, generator, device, iterations=4000, lr=0.1, real=False):
    optimizer = torch.optim.SGD(params=classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    it = iter(dataloader)
    loss_tot = 0.

    for i in range(iterations):
        it, batch = get_batch(it, dataloader)
        images, labels = batch[0].to(device), batch[1].to(device)

        if real:
            feat, lbl = get_real_features(model, images, labels)
        else:
            real_feat, real_lbl = get_real_features(model, images, labels)
            masked_feat, masked_lbl, real_feat, real_lbl = mask_features(real_feat, real_lbl)
            feat = generator(masked_feat).detach()
            lbl = masked_lbl

        score = classifier(feat)
        loss = criterion(score, lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tot += loss.item()

        if (i+1) % 10 == 0:
            print(f"Iter {i+1}: {loss_tot/10:.4f}")
            loss_tot = 0.


def val(model, device, loader, metrics):
    """Do validation and return specified samples"""
    metrics.reset()

    with torch.no_grad():
        model.eval()
        for i, (images, labels) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)  # B, C, H, W

            _, prediction = outputs.max(dim=1)  # B, H, W
            labels = labels.cpu().numpy()
            prediction = prediction.cpu().numpy()
            metrics.update(labels, prediction)


def main(opts):
    device = torch.device(opts.device) if opts.device is not None else "cuda"
    print(f"Device: {device}")

    task = Task(opts)
    task.disjoint = False
    task_name = f"{opts.task}-{opts.dataset}"

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst, train_dst_no_aug = get_dataset(opts, task, train=True)
    print(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    train_loader = data.DataLoader(train_dst, batch_size=10, shuffle=True,
                                   num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(val_dst, batch_size=10, shuffle=False,
                                 num_workers=opts.num_workers, pin_memory=True)

    step_ckpt = get_step_ckpt(opts, task_name)

    if opts.gen_pixtopix:
        generator = GlobalGenerator(128, 2048, ngf=64, n_downsampling=2, n_blocks=3,
                                    norm_layer=partial(nn.InstanceNorm2d, affine=False)).to(device)
    else:
        generator = FeatGenerator(128, 2048).to(device)
    generator.load_state_dict(step_ckpt['model_state']['generator'])

    classifier = CosineClassifier(task.get_n_classes(), channels=opts.n_feat)
    model = make_model(opts, classifier).to(device)
    model.load_state_dict(step_ckpt['model_state']['model'])

    del step_ckpt

    val_metrics = StreamSegMetrics(len(task.get_order()), task.get_n_classes()[0])

    new_classifier = nn.Sequential(
        DeeplabV3(2048, 256, 256,
                  norm_act=partial(InPlaceABN, activation="leaky_relu", activation_param=.01),
                  out_stride=opts.output_stride, pooling_size=opts.pooling,
                  pooling=not opts.no_pooling, last_relu=opts.relu),
        CosineClassifier(task.get_n_classes(), channels=opts.n_feat)
    )

    train(train_loader, model.eval(), new_classifier.train(), generator.eval(), device, iterations=4000, lr=0.1)

    model.head = new_classifier[0]
    model.cls = new_classifier[1]

    val(model, device, val_loader, val_metrics)

    val_score = val_metrics.get_results()
    print(val_metrics.to_str(val_score))


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    main(opts)
