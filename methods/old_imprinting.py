import torch
from .trainer import Trainer
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_module import make_model
import numpy as np
from modules.classifier import CosineClassifier
from torch.utils import data
import random
from .utils import get_batch, get_prototype
import math


class WeightMixing(Trainer):
    use_bkg = False

    def warm_up_(self, dataset, epochs=1):
        model = self.model.module if self.distributed else self.model
        model.eval()
        start_from = 0 if WeightMixing.use_bkg else 1
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = list(range(old_classes, old_classes + classes[-1]))
        for c in new_classes:
            ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
            images = [ds[i][0].unsqueeze(0) for i in range(len(ds))]
            labels = [ds[i][1] for i in range(len(ds))]
            with torch.no_grad():
                out = []
                for x in images:
                    x = x.to(self.device)
                    out.append(model(x).squeeze(0))
                oi_acc = torch.zeros(classes[0] - start_from).to(self.device)
                count = 0
                for i in range(len(out)):
                    labeli = labels[i].to(self.device)
                    if (labeli == c).sum() == 0:
                        continue
                    oi = out[i][:, labeli == c]  # get scores for pixels of c only -> oi has size C x P_c
                    oi = oi[start_from:classes[0]].softmax(dim=0)  # make softmax on base classes -> C_base x P_c
                    oi_acc += oi.mean(dim=1)
                    count += 1
                old_class_score = oi_acc / count  # now we have mean over all images
                new_weight = torch.zeros(self.n_channels).to(self.device)
                for oc in range(start_from, classes[0]):  # for each old class oc
                    new_weight += model.cls.cls[0].weight[oc].view(-1) * old_class_score[oc - start_from]
                model.cls.imprint_weights_class(new_weight.view(self.n_channels, 1, 1), c)


class ContextWiseWeightImprintingModule(nn.Module):
    def __init__(self, n_channels):
        super(ContextWiseWeightImprintingModule, self).__init__()
        self.dim = n_channels
        self.weight_c = nn.Parameter(F.normalize(torch.ones((self.dim, 1, 1)), dim=0))
        self.weight_b = nn.Parameter(-F.normalize(torch.ones((self.dim, 1, 1)), dim=0))

    def forward(self, cls_proto, bkg_proto):
        # input is a 2xD dimensional prototype
        weight = self.weight_c * cls_proto.mean(dim=0).view(-1, 1, 1)
        weight += self.weight_b * bkg_proto.mean(dim=0).view(-1, 1, 1)
        return weight.view(-1, 1, 1)


class ContextWiseWeightImprinting(Trainer):
    BATCH_SIZE = 4
    EPISODE = 2
    LR = 0.1
    ITER = 1

    def __init__(self, task, device, logger, opts):
        super(ContextWiseWeightImprinting, self).__init__(task, device, logger, opts)
        self.dim = self.n_channels

        self.weights = ContextWiseWeightImprintingModule(self.dim).to(self.device)

        self.LR = opts.dyn_lr
        self.ITER = opts.dyn_iter
        self.EPISODE = ContextWiseWeightImprinting.EPISODE
        self.BATCH_SIZE = ContextWiseWeightImprinting.BATCH_SIZE

    def warm_up_(self, dataset, epochs=1):
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        for c in new_classes:
            weight = None
            count = 0
            while weight is None and count < 10:
                ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
                protos = get_prototype(model, ds, c, self.device,
                                       interpolate_label=False, return_all=True, background=True)
                if protos is not None:
                    pc, pb = protos
                    weight = self.weights(pc, pb)
                count += 1

            if weight is not None:
                model.cls.imprint_weights_class(weight, c)
            else:
                raise Exception(f"Unable to imprint weight of class {c} after {count} trials.")

    def cool_down(self, dataset, epochs=1):
        if self.step == 0:
            # instance a new model without DDP!
            model = make_model(self.opts, CosineClassifier(self.task.get_n_classes(), channels=self.n_channels))
            scaler = model.cls.scaler
            state = {}
            for k, v in self.model.state_dict().items():
                state[k[7:]] = v
            model.load_state_dict(state, strict=True)
            model = model.to(self.device)
            model.fix_bn()
            model.eval()
            # instance optimizer, criterion and data
            classes = np.arange(0, self.task.get_n_classes()[0])
            classes = classes[1:]  # remove bkg ONLY from sampling
            params = [{"params": self.weights.parameters()}]
            optimizer = torch.optim.SGD(params, lr=self.LR)
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

            dataloader = data.DataLoader(dataset, batch_size=min(self.BATCH_SIZE, len(dataset)), shuffle=True,
                                         num_workers=4, drop_last=True)
            it = iter(dataloader)
            loss_tot = 0.
            # start train loop
            for i in range(self.ITER):
                loss_step = 0.
                optimizer.zero_grad()

                for e in range(self.EPISODE):
                    weight = torch.zeros_like(model.cls.cls[0].weight)
                    K = random.choice([1, 2, 5])
                    cls = random.choice(classes)  # sample N classes
                    for c in range(self.task.get_n_classes()[0]):
                        if c == cls:
                            ds = dataset.get_k_image_of_class(cl=cls, k=K)  # get K images of class c
                            protos = get_prototype(model, ds, cls, self.device,
                                                   interpolate_label=False, return_all=True, background=True)
                            if protos is None:
                                # print("WC is None!!")
                                weight[c] = F.normalize(model.cls.cls[0].weight[c], dim=0)
                            else:
                                pc, pb = protos
                                class_weight = self.weights(pc, pb)
                                weight[c] = F.normalize(class_weight, dim=0)
                        else:
                            weight[c] = F.normalize(model.cls.cls[0].weight[c], dim=0)

                    # get a batch of images from dataloader
                    it, batch = get_batch(it, dataloader)
                    ds = dataset.get_k_image_of_class(cl=cls, k=self.BATCH_SIZE)  # get K images of class c
                    im_ds = [ds[i][0].unsqueeze(0) for i in range(len(ds))]
                    lbl_ds = [ds[i][1].unsqueeze(0) for i in range(len(ds))]
                    images = torch.cat([batch[0], *im_ds], dim=0).to(self.device, dtype=torch.float32)
                    labels = torch.cat([batch[1], *lbl_ds], dim=0).to(self.device, dtype=torch.long)

                    with torch.no_grad():
                        out = model(images, use_classifier=False)
                        out = F.normalize(out, dim=1)
                    out = scaler * F.conv2d(out, weight)
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
                    loss = criterion(out, labels)
                    assert not torch.isnan(loss), "Error, loss is NaN"
                    loss.backward()
                    loss_step += loss.item()
                optimizer.step()

                self.logger.add_scalar("loss_cool_down", loss_step / self.EPISODE, i + 1)
                loss_tot += loss_step / self.EPISODE
                if ((i+1) % 10) == 0:
                    self.logger.info(f"Cool down loss at iter {i + 1}: {loss_tot / 10}")
                    loss_tot = 0

            state = {}
            if self.distributed:
                for k, v in model.state_dict().items():
                    state["module." + k] = v
            else:
                state = model.state_dict()
            self.model.load_state_dict(state)

    def load_state_dict(self, checkpoint, strict=True):
        super().load_state_dict(checkpoint, strict)
        if self.step > 0:
            self.weights.load_state_dict(checkpoint['weights'])
            self.weights.to(self.device)

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(),
                 "weights": self.weights.state_dict()}
        return state


class SpatialWeightGeneratorModule(nn.Module):
    def __init__(self, dim, naive=False):
        super(SpatialWeightGeneratorModule, self).__init__()
        self.dim = dim
        self.weight_q = nn.Linear(self.dim, self.dim, bias=False)
        self.weight_q.weight.data.copy_(torch.eye(self.dim, self.dim) + torch.randn(self.dim, self.dim) * 0.001)
        self.weight_k = nn.Linear(self.dim, self.dim, bias=False)
        self.weight_k.weight.data.copy_(torch.eye(self.dim, self.dim) + torch.randn(self.dim, self.dim) * 0.001)
        self.weight_v = nn.Linear(self.dim, self.dim, bias=False)
        self.weight_v.weight.data.copy_(torch.eye(self.dim, self.dim) + torch.randn(self.dim, self.dim) * 0.001)

        self.naive = naive

    def forward(self, images, labels, model, cls, device):
        # I expect feat with size BxCxHxW and lbl with size BxHxW
        if not self.training:
            with torch.no_grad():
                feat = []
                lbls = []
                for img, lbl in zip(images, labels):
                    f = model(img.to(device), use_classifier=False).detach()
                    f = F.interpolate(f, size=lbl.shape[-2:], mode="bilinear", align_corners=False)
                    f = f.permute(0, 2, 3, 1).flatten(end_dim=2)  # HW x C
                    feat.append(f)
                    lbls.append(lbl.flatten())  # HW

                feat = torch.cat(feat, dim=0)
                lbl = torch.cat(lbls, dim=0).to(device)
                mask = lbl.eq(cls).type(torch.long).view(-1, 1)
        else:
            img = torch.cat(images, dim=0).to(device)
            lbl = torch.cat(labels, dim=0).to(device)
            with torch.no_grad():
                feat = model(img, use_classifier=False).detach()
            out_size = feat.shape[-2:]

            feat = feat.permute(0, 2, 3, 1).flatten(end_dim=2)  # N x C
            lbl = (F.interpolate(lbl.float().unsqueeze(1),
                                 size=out_size, mode="nearest").type(torch.uint8))
            # Flat them all
            lbl = lbl.flatten()  # HW
            mask = lbl.eq(cls).type(torch.long)  # HW
            mask = mask.view(-1, 1)  # BHW = N

        feat = feat[lbl != 255]
        mask = mask[lbl != 255]

        if mask.sum() == 0:
            return None
        p_cls = (mask * feat).sum(dim=0, keepdim=True) / mask.sum()  # 1 x C
        if self.naive:
            Q = feat
            K = p_cls
            V = feat
        else:
            Q = self.weight_q(feat)  # N x C
            K = self.weight_k(p_cls)  # 1 x C
            V = self.weight_v(feat)  # N x C

        a_cl = F.softmax(K @ Q.T, dim=1)  # 1 x N
        weight = a_cl @ V
        return weight.view(-1, 1, 1)


class SpatialWeightGenerator(Trainer):  # SWG

    def __init__(self, task, device, logger, opts):
        super(SpatialWeightGenerator, self).__init__(task, device, logger, opts)
        self.dim = self.n_channels
        self.naive = False
        self.weights = SpatialWeightGeneratorModule(self.dim, self.naive).to(self.device)

        self.LR = opts.dyn_lr
        self.ITER = opts.dyn_iter
        self.EPISODE = 2
        self.BATCH_SIZE = 4

    def warm_up_(self, dataset, epochs=1):
        model = self.model.module if self.distributed else self.model
        model.eval()
        self.weights.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        for c in new_classes:
            weight = None
            count = 0
            while weight is None and count < 10:
                ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
                img = [ds[i][0].unsqueeze(0) for i in range(len(ds))]
                lbl = [ds[i][1].unsqueeze(0) for i in range(len(ds))]
                weight = self.weights(img, lbl, model, c, self.device)
                count += 1

            if weight is not None:
                model.cls.imprint_weights_class(weight, c)
            else:
                raise Exception(f"Unable to imprint weight of class {c} after {count} trials.")

    def cool_down(self, dataset, epochs=1):
        if self.step == 0:
            # instance a new model without DDP!
            model = make_model(self.opts, CosineClassifier(self.task.get_n_classes(), channels=self.n_channels))
            scaler = model.cls.scaler
            self.weights.train()
            state = {}
            for k, v in self.model.state_dict().items():
                state[k[7:]] = v
            model.load_state_dict(state, strict=True)
            model = model.to(self.device)
            model.fix_bn()
            model.eval()
            # instance optimizer, criterion and data
            classes = np.arange(0, self.task.get_n_classes()[0])
            classes = classes[1:]  # remove bkg ONLY from sampling
            params = [{"params": self.weights.parameters()}]
            optimizer = torch.optim.SGD(params, lr=self.LR)
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

            dataloader = data.DataLoader(dataset, batch_size=min(self.BATCH_SIZE, len(dataset)), shuffle=True,
                                         num_workers=4, drop_last=True)
            it = iter(dataloader)
            loss_tot = 0.
            # start train loop
            for i in range(self.ITER):
                loss_step = 0.
                optimizer.zero_grad()

                for e in range(self.EPISODE):
                    weight = torch.zeros_like(model.cls.cls[0].weight)
                    K = random.choice([1, 2, 5])
                    cls = random.choice(classes)  # sample N classes
                    for c in range(self.task.get_n_classes()[0]):
                        if c == cls:
                            ds = dataset.get_k_image_of_class(cl=cls, k=K)  # get K images of class c
                            img = [ds[i][0].unsqueeze(0) for i in range(len(ds))]
                            lbl = [ds[i][1].unsqueeze(0) for i in range(len(ds))]
                            class_weight = self.weights(img, lbl, model, c, self.device)

                            if class_weight is None:
                                # print("WC is None!!")
                                weight[c] = F.normalize(model.cls.cls[0].weight[c], dim=0)
                            else:
                                weight[c] = F.normalize(class_weight, dim=0)
                        else:
                            weight[c] = F.normalize(model.cls.cls[0].weight[c], dim=0)

                    # get a batch of images from dataloader
                    it, batch = get_batch(it, dataloader)
                    ds = dataset.get_k_image_of_class(cl=cls, k=self.BATCH_SIZE)  # get K images of class c
                    im_ds = [ds[i][0].unsqueeze(0) for i in range(len(ds))]
                    lbl_ds = [ds[i][1].unsqueeze(0) for i in range(len(ds))]
                    images = torch.cat([batch[0], *im_ds], dim=0).to(self.device, dtype=torch.float32)
                    labels = torch.cat([batch[1], *lbl_ds], dim=0).to(self.device, dtype=torch.long)

                    with torch.no_grad():
                        out = model(images, use_classifier=False)
                        out = F.normalize(out, dim=1)
                    out = scaler * F.conv2d(out, weight)
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
                    loss = criterion(out, labels)
                    assert not torch.isnan(loss), "Error, loss is NaN"
                    loss.backward()
                    loss_step += loss.item()
                optimizer.step()

                self.logger.add_scalar("loss_cool_down", loss_step / self.EPISODE, i + 1)
                loss_tot += loss_step / self.EPISODE
                if ((i+1) % 10) == 0:
                    self.logger.info(f"Cool down loss at iter {i + 1}: {loss_tot / 10}")
                    loss_tot = 0

            state = {}
            if self.distributed:
                for k, v in model.state_dict().items():
                    state["module." + k] = v
            else:
                state = model.state_dict()
            self.model.load_state_dict(state)

    def load_state_dict(self, checkpoint, strict=True):
        super().load_state_dict(checkpoint, strict)
        if self.step > 0 and not self.naive:
            self.weights.load_state_dict(checkpoint['weights'])
            self.weights.to(self.device)

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(),
                 "weights": self.weights.state_dict()}
        return state
