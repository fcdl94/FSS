import torch
from .trainer import Trainer
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_module import make_model
import numpy as np
from modules.classifier import CosineClassifier
from torch.utils import data
import random
from modules import DeeplabV3
from .utils import get_batch, get_prototype, myReLU
from utils.loss import HardNegativeMining


class AMP(Trainer):
    EPOCHS = 5

    def initialize(self, opts):
        super(AMP, self).initialize(opts)
        self.amp_alpha = opts.amp_alpha

    def warm_up(self, dataset, epochs=1):
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = len(self.task.order)
        sum_features = torch.zeros(classes, model.head_channels).to(self.device)
        count_features = torch.zeros(classes).to(self.device)

        for ep in range(AMP.EPOCHS):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    out_size = images.shape[-2:]
                    out = model(images.unsqueeze(0), use_classifier=False)
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
                model.cls.imprint_weights_class(features=features, cl=c, alpha=self.amp_alpha)


class WeightImprinting(Trainer):
    def __init__(self, task, device, logger, opts):
        super().__init__(task, device, logger, opts)
        self.pixel = opts.pixel_imprinting
        if opts.weight_mix:
            self.normalize_weight = True
            self.compute_score = True
        else:
            self.normalize_weight = False
            self.compute_score = False

    def warm_up(self, dataset, epochs=5):
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        sum_features = torch.zeros(classes[-1], model.head_channels).to(self.device)
        oi_acc = torch.zeros(classes[-1], classes[0] - 1).to(self.device)
        count = torch.zeros(classes[-1]).to(self.device)
        for ep in range(epochs):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    out = model(images.unsqueeze(0), use_classifier=False)  # .squeeze_(0)
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False).squeeze_(0)
                    # labels = F.interpolate(labels.float().view(1, 1, labels.shape[0], labels.shape[1]),
                    #                        size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
                    cl = labels.unique().cpu().numpy()  # get list of classes
                    for c in new_classes:
                        if c in cl:
                            feat = out[:, labels == c]  # F x P (pixels of class c)

                            if self.compute_score:
                                oi = model.cls.cls[0](out.unsqueeze(0)).squeeze(0)
                                oi = oi[:, labels == c]  # get scores for pixels of c only -> oi has size C x P_c
                                oi = oi[1:].softmax(dim=0)  # make softmax on base classes / bkg -> C_b x P_c
                                oi_acc[c - old_classes] += oi.mean(dim=1)

                            if self.pixel:
                                feat = F.normalize(feat, dim=0).t()
                                p = model.cls.cls[0].weight[0].view(1, 256).detach()
                                # p = p / p.norm()
                                bkg_score = -(feat * p).sum(dim=1)
                                max_score, max_idx = bkg_score.max(dim=0)
                                feat = feat[max_idx]
                            else:
                                feat = F.normalize(F.normalize(feat, dim=0).sum(dim=1), dim=0)
                            sum_features[c - old_classes] += feat
                            count[c - old_classes] += 1

        # we have finished computing features, now collect and imprint!
        assert torch.any(count != 0), "Error, a novel class has no pixels!"
        features = F.normalize(sum_features, dim=1)
        if self.normalize_weight:
            if self.compute_score:
                scores = oi_acc / count.view(-1, 1)
                for c in range(classes[-1]):
                    f = (model.cls.cls[0].weight[1:] * scores[c].view(classes[0] - 1, 1, 1, 1))
                    features[c] += f.sum(dim=0).view(model.head_channels)
            else:
                features += model.cls.cls[0].weight[1:].mean(dim=0).view(-1).detach()
        model.cls.imprint_weights_step(step=self.task.step, features=features)


class WeightMixing(Trainer):
    use_bkg = False

    def warm_up(self, dataset, epochs=1):
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


class DynamicWI(Trainer):
    LR = 0.01
    ITER = 500
    BATCH_SIZE = 10
    EPISODE = 5

    def __init__(self, task, device, logger, opts):
        super(DynamicWI, self).__init__(task, device, logger, opts)
        self.weight = nn.Parameter(F.normalize(torch.ones((self.n_channels, 1, 1), device=self.device), dim=0))

    def warm_up(self, dataset, epochs=1):
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        for c in new_classes:
            ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
            wc = get_prototype(model, ds, c, self.device)
            count = 0
            while wc is None and count < 10:
                ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
                wc = get_prototype(model, ds, c, self.device)
                count += 1
            model.cls.imprint_weights_class(F.normalize(self.weight * wc.view(self.n_channels, 1, 1), dim=0), c)

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
            classes = classes[1:] if self.task.use_bkg else classes  # remove bkg if present
            params = [  # {"params": model.cls.cls[0].weight, "lr": DynamicWI.LR*0.1},
                {"params": self.weight, "lr": DynamicWI.LR}]
            optimizer = torch.optim.SGD(params, lr=DynamicWI.LR)
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

            dataloader = data.DataLoader(dataset, batch_size=min(DynamicWI.BATCH_SIZE, len(dataset)), shuffle=True,
                                         num_workers=4, drop_last=True)
            it = iter(dataloader)
            loss_tot = 0.
            # start train loop
            for i in range(DynamicWI.ITER):
                loss_step = 0.
                optimizer.zero_grad()

                for e in range(DynamicWI.EPISODE):
                    weight = torch.zeros_like(model.cls.cls[0].weight)
                    K = random.choice([1, 5, 10])
                    cls = random.choice(classes)  # sample N classes
                    for c in range(self.task.get_n_classes()[0]):
                        wc = None
                        if c == cls:
                            ds = dataset.get_k_image_of_class(cl=c, k=K)  # get K images of class c
                            wc = get_prototype(self.model, ds, c, self.device)
                        if wc is None:
                            weight[c] = F.normalize(model.cls.cls[0].weight[c], dim=0)
                        else:
                            weight[c] = F.normalize(self.weight * wc.view(self.n_channels, 1, 1), dim=0)

                    # get a batch of images from dataloader
                    it, batch = get_batch(it, dataloader)
                    ds = dataset.get_k_image_of_class(cl=cls, k=DynamicWI.BATCH_SIZE)  # get K images of class c
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
                    loss.backward()
                    loss_step += loss.item()
                optimizer.step()

                self.logger.add_scalar("loss_cool_down", loss_step / DynamicWI.EPISODE, i + 1)
                loss_tot += loss_step / DynamicWI.EPISODE
                if (i % 50) == 0:
                    self.logger.info(f"Cool down loss at iter {i + 1}: {loss_tot / (i + 1)}")

            self.logger.debug(self.weight)
            state = {}
            for k, v in model.state_dict().items():
                state["module." + k] = v
            self.model.load_state_dict(state)

    def load_state_dict(self, checkpoint, strict=True):
        super().load_state_dict(checkpoint, strict)
        if self.step > 0:
            device = self.weight.device
            self.weight.data = checkpoint['weight'].to(device)

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "weight": self.weight.data}
        return state


class WeightGenerator(Trainer):
    def __init__(self, task, device, logger, opts):
        super().__init__(task, device, logger, opts)

        self.weight_gen = DeeplabV3(self.model.body.out_channels, self.n_channels, 256,
                                    norm_act=myReLU,
                                    out_stride=opts.output_stride, pooling_size=opts.pooling,
                                    pooling=not opts.no_pooling, last_relu=False).to(self.device)
        self.weight_feat = nn.Linear(256, 256, bias=False).to(self.device)
        self.LR = opts.lr
        self.ITER = opts.iter
        self.BATCH_SIZE = 10
        self.EPISODE = 5
        self.train = True

    def gen_weight(self, model, ds, cl):
        self.weight_gen.eval()

        if self.train:
            im_ds = [ds[i][0].unsqueeze(0) for i in range(len(ds))]
            lbl_ds = [ds[i][1].unsqueeze(0) for i in range(len(ds))]
            images = torch.cat(im_ds, dim=0).to(self.device, dtype=torch.float32)
            labels = torch.cat(lbl_ds, dim=0).to(self.device, dtype=torch.long)
            out, feat = model(images, body_feat=True)
            mask = F.adaptive_avg_pool2d((labels == cl).float().unsqueeze(1), output_size=feat.shape[-2:])
            feat = F.max_pool2d(feat * mask,
                                kernel_size=feat.size()[-2:])  # this is equal to make ReLU then MaxPool -> BxF
            x = self.weight_gen(feat)  # now feat are B x F x H x W
            w = self.weight_feat(x.view(len(images), -1))  # pass into linear layer -> BxF
        else:
            w = []
            for i in range(len(ds)):
                images, labels = ds[i]
                images = images.to(self.device, dtype=torch.float32).unsqueeze(0)
                labels = labels.to(self.device, dtype=torch.long).unsqueeze(0)
                out, feat = model(images, body_feat=True)
                mask = F.adaptive_avg_pool2d((labels == cl).float().unsqueeze(1), output_size=feat.shape[-2:])
                feat = F.max_pool2d(feat * mask,
                                    kernel_size=feat.size()[-2:])  # this is equal to make ReLU then MaxPool -> BxF
                x = self.weight_gen(feat)  # now feat are B x F x H x W
                w.append(self.weight_feat(x.view(len(images), -1)))  # pass into linear layer -> 1xF
            w = torch.cat(w, dim=0)

        w = w.mean(dim=0)  # F

        return w

    def warm_up(self, dataset, epochs=1):
        self.train = False
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        for c in new_classes:
            ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
            wc = self.gen_weight(model, ds, c)  # generated weigth
            model.cls.imprint_weights_class(wc.view(self.n_channels, 1, 1), c)

    def init_as_model(self, state):
        self.weight_gen.load_state_dict(state, strict=False)

    def cool_down(self, dataset, epochs=1):
        if self.step == 0:
            self.train = True
            # instance a new model without DDP!
            model = self.make_model()
            scaler = model.cls.scaler
            state = {}
            for k, v in self.model.state_dict().items():
                state[k[7:]] = v
            model.load_state_dict(state, strict=True)
            model = model.to(self.device)
            model.eval()
            # self.init_as_model(model.body.state_dict())
            # instance optimizer, criterion and data
            classes = np.arange(0, self.task.get_n_classes()[0])
            classes = classes[1:] if self.task.use_bkg else classes  # remove bkg if present
            params = [{"params": self.weight_gen.parameters(), "lr": self.LR},
                      {"params": self.weight_feat.parameters(), "lr": self.LR}]
            optimizer = torch.optim.SGD(params, lr=self.LR)
            ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
            hnm = HardNegativeMining()

            def criterion(x, y):
                return hnm(ce_loss(x, y))

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
                    K = random.choice([1, 5, 10])
                    cls = random.choice(classes)  # sample N classes
                    for c in range(self.task.get_n_classes()[0]):
                        wc = None
                        if c == cls:
                            ds = dataset.get_k_image_of_class(cl=c, k=K)  # get K images of class c
                            wc = self.gen_weight(model, ds, cls)  # generated weigth
                        if wc is None:
                            weight[c] = F.normalize(model.cls.cls[0].weight[c], dim=0)
                        else:
                            weight[c] = wc.view(self.n_channels, 1, 1)

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
                    out = scaler * F.conv2d(out, F.normalize(weight, dim=1))
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
                    loss = criterion(out, labels)
                    loss.backward()
                    loss_step += loss.item()
                optimizer.step()

                self.logger.add_scalar("loss_cool_down", loss_step / self.EPISODE, i + 1)
                loss_tot += loss_step / self.EPISODE
                if (i % 50) == 0:
                    self.logger.info(f"Cool down loss at iter {i + 1}: {loss_tot / (i + 1)}")
                if (i % 500) == 0:
                    state = {"model_state": self.state_dict()}
                    opts = self.opts
                    torch.save(state, f"checkpoints/step/{opts.task}-{opts.dataset}/{opts.name}_{opts.step}.pth")

    def load_state_dict(self, checkpoint, strict=True):
        super().load_state_dict(checkpoint, strict)
        if self.step > 0:
            assert "weight_gen" in checkpoint, "Pretrained model does not have weights for generator."
        if "weight_gen" in checkpoint:
            self.logger.info("Restoring weight generator state.")
            self.weight_gen.load_state_dict(checkpoint["weight_gen"])
            self.weight_feat.load_state_dict(checkpoint["weight_feat"])

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "weight_gen": self.weight_gen.state_dict(),
                 "weight_feat": self.weight_feat.state_dict()}
        return state
