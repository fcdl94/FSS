import torch
from .trainer import Trainer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from .utils import get_batch
from functools import partial
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import numpy as np
from modules.generators import FeatGenerator, DCGANDiscriminator, GlobalGenerator, GlobalGenerator2
from modules.deeplab import DeeplabV3
from inplace_abn import InPlaceABN
from modules.classifier import CosineClassifier
from torch.distributions import normal
from utils.loss import BinaryCrossEntropy


class FGI(Trainer):
    def __init__(self, task, device, logger, opts):
        super().__init__(task, device, logger, opts)

        self.dim = 2048
        self.z_dim = 128
        self.cond_gan = opts.gen_cond_gan
        self.n_classes = task.get_n_classes()[0]
        self.Z_dist = normal.Normal(0, 1)

        if not opts.type2:
            self.z_dim = 128
            self.in_dim = 2048
            self.mask_func = self.mask_features1
            self.add_Z = True
        else:
            self.z_dim = 0
            self.in_dim = 2049
            self.mask_func = self.mask_features2
            self.add_Z = False

        if opts.gen_pixtopix:
            self.generator = GlobalGenerator(self.z_dim, self.in_dim, self.dim, ngf=opts.ngf, n_downsampling=2, n_blocks=3,
                                             norm_layer=partial(nn.InstanceNorm2d, affine=False)).to(device)
        else:
            self.generator = FeatGenerator(self.z_dim, self.in_dim, self.dim, n_layer=opts.gen_nlayer).to(device)
        if self.cond_gan:
            self.discriminator = nn.Sequential(
                DeeplabV3(2048, 256, 256,
                          norm_act=partial(InPlaceABN, activation="leaky_relu", activation_param=.01),
                          out_stride=opts.output_stride, pooling_size=opts.pooling,
                          pooling=not opts.no_pooling, last_relu=opts.relu),
                CosineClassifier([self.n_classes+1], channels=opts.n_feat)
            ).to(device)
        else:
            self.discriminator = DCGANDiscriminator(in_feat=self.dim+1, dim=256).to(device)
        self.discriminator.train()

        self.ITER = opts.gen_iter
        self.LR = opts.gen_lr
        self.BATCH_SIZE = 10
        self.n_critic = opts.gen_ncritic
        self.lmbda = 10
        self.alpha = opts.gen_alpha
        self.beta = 1
        self.use_cls_loss = True
        self.use_bkg_loss = opts.gen_use_bkg_loss
        self.gen_mib = opts.gen_mib

        if task.step > 0:
            self.generated_criterion = nn.CrossEntropyLoss(reduction='mean')
            self.generator.eval()
        else:
            self.starting_iter = 0
            self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.LR)
            self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.LR)

    def warm_up_(self, dataset, epochs=1):
        model = self.model.module if self.distributed else self.model
        model.eval()
        classes = self.task.get_n_classes()
        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        new_classes = np.arange(old_classes, old_classes + classes[-1])
        sum_features = torch.zeros(classes[-1], model.head_channels).to(self.device)
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

                            feat = F.normalize(F.normalize(feat, dim=0).sum(dim=1), dim=0)
                            sum_features[c - old_classes] += feat
                            count[c - old_classes] += 1

        # we have finished computing features, now collect and imprint!
        assert torch.any(count != 0), "Error, a novel class has no pixels!"
        features = F.normalize(sum_features, dim=1)
        model.cls.imprint_weights_step(step=self.task.step, features=features)

    def generative_loss(self, images=None, labels=None):
        with torch.no_grad():
            gen_feat, gen_target = self.generate_synth_feat(images, labels)
        score = self.model(gen_feat, only_head=True)
        loss = self.generated_criterion(score, gen_target)

        if self.model_old is not None:
            score_old = self.model_old(gen_feat, only_head=True)
            if self.kd_criterion is not None:
                loss += self.kd_loss * self.kd_criterion(score, score_old)

        return self.gen_weight * loss

    def generate_synth_feat(self, images=None, labels=None):
        real_feat, real_lbl = self.get_real_features(self.model, images, labels)
        masked_feat, masked_lbl, real_feat, real_lbl = self.mask_func(real_feat, real_lbl)

        gen_feat = self.generator(masked_feat).detach()

        return gen_feat, masked_lbl

    @staticmethod
    def get_real_features(model, images, labels):
        with torch.no_grad():
            _, _, feat = model(images, return_feat=True, return_body=True)
        # Downsample labels to match feat size (32x32)
        labels = F.interpolate(labels.float().unsqueeze(1), size=feat.shape[-2:], mode="nearest").long()

        return feat, labels.squeeze(1)  # get back to B x H x W

    @staticmethod
    def get_bkg_proto(feat, labels):
        protos = []
        for i in range(feat.shape[0]):  # each image independently
            mask = labels[i] == 0
            if mask.sum() > 0:
                f = feat[i][:, mask].mean(dim=1)  # should  be 1xD
                protos.append(f.view(1, -1))
        return torch.cat(protos, dim=0)

    def mask_features1(self, feat, lbl):
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

    def mask_features2(self, feat, lbl):
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
                z = self.Z_dist.sample(feat[i].shape).to(self.device)

                fz = (feat[i] * m.float() + z * (-m.float() + 1))
                mf = torch.cat((fz, m.unsqueeze(0).float()), dim=0)
                mask_feat.append(mf.unsqueeze(0))

                real_feat.append(feat[i].unsqueeze(0))
                mask_lbl.append((lbl[i] * m.long()).unsqueeze(0))
                real_lbl.append((lbl[i]).unsqueeze(0))
        return torch.cat(mask_feat, dim=0), torch.cat(mask_lbl, dim=0).long(), \
               torch.cat(real_feat, dim=0), torch.cat(real_lbl, dim=0).long()

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.shape[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(self.device)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()

    def cool_down(self, dataset, epochs=1):
        if self.step == 0:
            # instance a new model without DDP!
            model = self.model if not self.distributed else self.model.module
            model.fix_bn()
            model.eval()

            if self.cond_gan:  # Initialize GAN discriminator with head parameters
                self.discriminator[0].load_state_dict(model.head.state_dict())

            # instance optimizer, criterion and data
            optim_G = self.optim_G
            optim_D = self.optim_D

            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            if self.gen_mib:
                criterion_gen = BinaryCrossEntropy()
            else:
                criterion_gen = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

            dataloader = data.DataLoader(dataset, batch_size=min(self.BATCH_SIZE, len(dataset)), shuffle=True,
                                         num_workers=4, drop_last=True)

            it = iter(dataloader)
            loss_tot = 0.
            class_loss = 0.
            ar_loss = 0.

            for i in range(self.starting_iter, self.ITER):
                it, batch = get_batch(it, dataloader)
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                real_feat, real_lbl, = self.get_real_features(model, images, labels)
                masked_feat, masked_lbl, real_feat, real_lbl = self.mask_func(real_feat, real_lbl)
                mask = -(masked_lbl == 0).float() + 1
                mask = mask.unsqueeze(1)

                total_discr_loss = 0.
                total_grad_penalty = 0.
                get_tot_loss = 0.
                #
                # Optimize Critic (n times)
                #
                for _ in range(self.n_critic):
                    gen_feat = self.generator(masked_feat, add_z=self.add_Z)

                    if self.cond_gan:   # compute Cond GAN loss
                        discr_loss = criterion(self.discriminator(real_feat), real_lbl)
                        discr_loss += criterion(self.discriminator(gen_feat), torch.full_like(real_lbl, self.n_classes))
                        grad_penalty = 0.
                    else:  # calculate normal WGAN loss
                        rf = torch.cat((real_feat, mask), dim=1)
                        gf = torch.cat((gen_feat, mask), dim=1)
                        discr_loss = (self.discriminator(gf) - self.discriminator(rf)).mean()

                        # calculate gradient penalty
                        grad_penalty = self.lmbda * self._gradient_penalty(rf, gf)

                    # update critic params
                    optim_D.zero_grad()
                    (discr_loss + grad_penalty).backward()
                    optim_D.step()

                    total_grad_penalty += grad_penalty / self.n_critic
                    total_discr_loss += discr_loss.item() / self.n_critic

                #
                # Optimize Generator (n times)
                #
                gen_feat = self.generator(masked_feat, add_z=self.add_Z)

                if self.cond_gan:
                    gen_loss = criterion_gen(self.discriminator(gen_feat), masked_lbl)
                else:
                    gf = torch.cat((gen_feat, mask), dim=1)
                    gen_loss = - (torch.mean(self.discriminator(gf)))
                get_tot_loss += gen_loss

                if self.use_cls_loss:
                    classifier = nn.Sequential(model.head, model.cls)
                    score = classifier(gen_feat)
                    class_loss = criterion_gen(score, masked_lbl)
                    get_tot_loss += self.alpha * class_loss

                if self.use_bkg_loss:
                    bkg_real = self.get_bkg_proto(real_feat, masked_lbl)
                    bkg_fake = self.get_bkg_proto(gen_feat, masked_lbl)
                    ar_loss = F.cosine_similarity(bkg_real, bkg_fake)
                    ar_loss = ar_loss.mean()
                    get_tot_loss += self.beta * ar_loss.mean()

                optim_G.zero_grad()
                get_tot_loss.backward()
                optim_G.step()

                loss_step = get_tot_loss + total_discr_loss + total_grad_penalty

                self.logger.add_scalar("loss_cool_down", loss_step, i + 1)
                loss_tot += loss_step
                if ((i + 1) % 10) == 0:
                    self.logger.info(f"Average loss at iter {i + 1:4}: {loss_tot / 10 :8.4f} \n"
                                     f"\t Step {i+1}: G {gen_loss:8.4f} D {total_discr_loss:8.4f} "
                                     f" GP {total_grad_penalty:8.4f}"
                                     f" C {class_loss:8.4f} A {ar_loss:8.4f}")
                    loss_tot = 0

    def load_state_dict(self, checkpoint, strict=True):
        super().load_state_dict(checkpoint, strict)
        if self.step > 0:
            assert 'generator' in checkpoint, "Error, generator is not in checkpoint."
        if 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
            self.generator.to(self.device)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.discriminator.to(self.device)
            if "iter" in checkpoint:
                self.starting_iter = checkpoint['iter']
                self.optim_G.load_state_dict(checkpoint['optim_G'])
                self.optim_D.load_state_dict(checkpoint['optim_D'])

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(),
                 "generator": self.generator.state_dict(), "discriminator": self.discriminator.state_dict(),
                 "iter": self.ITER, "optim_G": self.optim_G.state_dict(), "optim_D": self.optim_D.state_dict()}
        return state


