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


class AMP(Trainer):
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


class WeightImprinting(Trainer):
    def warm_up(self, dataset, epochs=5):
        self.model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes+classes[-1])
        sum_features = torch.zeros(classes[-1], self.model.module.head_channels).to(self.device)
        count_features = torch.zeros(classes[-1]).to(self.device)
        for ep in range(epochs):
            with torch.no_grad():
                for idx in range(len(dataset)):
                    images, labels = dataset[idx]
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    out = self.model(images.unsqueeze(0), use_classifier=False)  # .squeeze_(0)
                    out_size = images.shape[-2:]
                    out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False).squeeze_(0)
                    # labels = F.interpolate(labels.float().view(1, 1, labels.shape[0], labels.shape[1]),
                    #                        size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
                    cl = labels.unique().cpu().numpy()  # get list of classes
                    for c in new_classes:
                        if c in cl:
                            feat = out[:, labels == c]  # F x P (pixels of class c)
                            sum_features[c-old_classes] += F.normalize(F.normalize(feat, dim=0).sum(dim=1), dim=0)
                            count_features[c-old_classes] += 1
        # we have finished computing features, now collect and imprint!
        assert torch.any(count_features != 0), "Error, a novel class has no pixels!"
        features = F.normalize(sum_features, dim=1)
        self.model.module.cls.imprint_weights_step(step=self.task.step, features=features)


class DynamicWI(Trainer):
    LR = 0.01
    ITER = 500
    BATCH_SIZE = 10
    EPISODE = 5

    def __init__(self, task, device, logger, opts):
        super(DynamicWI, self).__init__(task, device, logger, opts)
        self.weight = nn.Parameter(F.normalize(torch.ones((self.n_channels, 1, 1), device=self.device), dim=0))

    def warm_up(self, dataset, epochs=1):
        self.model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        for c in new_classes:
            ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
            wc = get_prototype(self.model, ds, c, self.device)
            count = 0
            while wc is None and count < 10:
                ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
                wc = get_prototype(self.model, ds, c, self.device)
                count += 1
            self.model.module.cls.imprint_weights_class(F.normalize(self.weight * wc.view(self.n_channels, 1, 1), dim=0), c)

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

                self.logger.add_scalar("loss_cool_down", loss_step/DynamicWI.EPISODE, i+1)
                loss_tot += loss_step/DynamicWI.EPISODE
                if (i % 50) == 0:
                    self.logger.info(f"Cool down loss at iter {i+1}: {loss_tot/(i+1)}")

            self.logger.debug(self.weight)
            state = {}
            for k, v in model.state_dict().items():
                state["module."+k] = v
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


class WeightMixing(Trainer):

    def warm_up(self, dataset, epochs=1):
        self.model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = list(range(old_classes, old_classes + classes[-1]))
        for c in new_classes:
            ds = dataset.get_k_image_of_class(cl=c, k=self.task.nshot)  # get K images of class c
            im_ds = [ds[i][0].unsqueeze(0) for i in range(len(ds))]
            lbl_ds = [ds[i][1].unsqueeze(0) for i in range(len(ds))]
            images = torch.cat(im_ds, dim=0).to(self.device, dtype=torch.float32)
            labels = torch.cat(lbl_ds, dim=0).to(self.device, dtype=torch.long)
            with torch.no_grad():
                out = self.model(images)
                oi_acc = torch.zeros(classes[0]).to(self.device)
                count = 0
                for i in range(len(out)):
                    if (labels[i] == c).sum() == 0:
                        continue
                    oi = out[i][:, labels[i] == c]  # get scores for pixels of c only -> oi has size C x P_c
                    oi = oi[:classes[0]].softmax(dim=0)  # make softmax on base classes -> C_base x P_c
                    oi_acc += oi.mean(dim=1)
                    count += 1
                old_class_score = oi_acc / count  # now we have mean over all images
                new_weight = torch.zeros(self.n_channels).to(self.device)
                for oc in range(classes[0]):  # for each old class oc
                    new_weight += self.model.module.cls.cls[0].weight[oc].view(-1) * old_class_score[oc]
                self.model.module.cls.imprint_weights_class(new_weight.view(self.n_channels, 1, 1), c)
