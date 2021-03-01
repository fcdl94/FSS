import torch
from .trainer import Trainer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from .utils import get_batch
from torch.distributions import normal
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import numpy as np


class Generator(nn.Module):
    def __init__(self, z_dim, attr_dim, out_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, out_dim),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim, attr_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(x_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class AFHN(Trainer):
    def __init__(self, task, device, logger, opts):
        super().__init__(task, device, logger, opts)

        self.dim = 256
        self.z_dim = 128
        self.generator = Generator(self.z_dim, self.dim, self.dim).to(device)
        self.discriminator = Discriminator(self.dim, self.dim).to(device)
        self.discriminator.train()
        self.Z_dist = normal.Normal(0, 1)

        if self.task.step > 0:
            for par in self.generator.parameters():
                par.requires_grad = False

        self.ITER = 10000
        self.BATCH_SIZE = 10
        self.gen_BS = 512
        self.n_critic = 5
        self.lmbda = 10
        self.alpha = 1
        self.beta = 1
        self.use_cls_loss = True
        self.use_anti_collapse = True

        classes = self.task.get_n_classes()

        self.class_seed = torch.zeros(self.task.num_classes, self.dim).to(self.device)

        old_classes = 0
        for c in classes[:-1]:
            old_classes += c
        self.labels = torch.LongTensor(np.arange(old_classes, old_classes + classes[-1]))
        if task.step > 0:
            self.generated_criterion = nn.CrossEntropyLoss(reduction='mean')

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
        with torch.no_grad():
            for idx in range(len(dataset)):
                images, labels = dataset[idx]
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                out = model(images.unsqueeze(0), use_classifier=False)

                out_size = images.shape[-2:]
                out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False).squeeze_(0)

                cl = labels.unique().cpu().numpy()  # get list of classes
                for c in new_classes:
                    if c in cl:
                        feat = out[:, labels == c]  # F x P (pixels of class c)
                        feat = feat.mean(dim=1)
                        sum_features[c - old_classes] += feat
                        count[c - old_classes] += 1

        # we have finished computing features, now collect and store for generation
        for c in new_classes:
            self.class_seed[c] = sum_features[c-old_classes] / count[c-old_classes]

    def update_means(self, real_feat, real_lbl):
        lbl = real_lbl.view(-1)
        for i, c in enumerate(lbl):
            self.class_seed[c] = 0.9 * self.class_seed[c] + 0.1 * real_feat[i]

    def generate_synth_feat(self, images=None, labels=None):
        labels = np.random.choice(np.arange(1, self.task.num_classes), self.gen_BS)
        feat = self.class_seed[labels]

        Z = self.Z_dist.sample((feat.shape[0], self.z_dim)).to(self.device)
        gen_in = torch.cat((Z, feat), dim=1)
        gen_feat = self.generator(gen_in).view(-1, self.dim, 1, 1).detach()
        gen_lbl = torch.tensor(labels).view(-1, 1, 1)

        return gen_feat, gen_lbl

    @staticmethod
    def get_real_features(model, images, labels):
        with torch.no_grad():
            feat = model(images, use_classifier=False)
        labels = F.interpolate(labels.float().unsqueeze(1),
                               size=feat.shape[-2:], mode="nearest").type(torch.uint8)
        feat = feat.permute(0, 2, 3, 1).flatten(end_dim=2)  # B H W C
        labels = labels.flatten()
        real_feat = []
        real_lbls = []
        for c in labels.unique():
            if c != 255:
                real_feat.append(feat[labels == c].mean(dim=0, keepdim=True))  # 1 x C
                real_lbls.append(c.view(1, 1))
        real_feat = torch.cat(real_feat, dim=0)
        real_lbls = torch.cat(real_lbls, dim=0)
        return real_feat, real_lbls

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
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
            # instance optimizer, criterion and data
            optim_G = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
            optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

            dataloader = data.DataLoader(dataset, batch_size=min(self.BATCH_SIZE, len(dataset)), shuffle=True,
                                         num_workers=4, drop_last=True)

            it = iter(dataloader)
            loss_tot = 0.
            class_loss = 0.
            ar_loss = 0.

            for i in range(self.ITER):
                it, batch = get_batch(it, dataloader)
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                real_feat, real_lbl = self.get_real_features(model, images, labels)

                self.update_means(real_feat, real_lbl)

                total_discr_loss = 0.
                total_grad_penalty = 0.
                get_tot_loss = 0.
                #
                # Optimize Critic (n times)
                #
                for _ in range(self.n_critic):
                    Z1 = self.Z_dist.sample((real_feat.shape[0], self.z_dim)).to(self.device)
                    gen_in1 = torch.cat((Z1, real_feat), dim=1)
                    gen_feat1 = self.generator(gen_in1)

                    Z2 = self.Z_dist.sample((real_feat.shape[0], self.z_dim)).to(self.device)
                    gen_in2 = torch.cat((Z2, real_feat), dim=1)
                    gen_feat2 = self.generator(gen_in2)

                    X_gen1 = torch.cat((gen_feat1, real_feat), dim=1)
                    X_gen2 = torch.cat((gen_feat2, real_feat), dim=1)
                    X_real = torch.cat((real_feat, real_feat), dim=1)

                    # calculate normal WGAN loss
                    discr_loss = 0.5*(self.discriminator(X_gen1) - self.discriminator(X_real)).mean()
                    discr_loss += 0.5*(self.discriminator(X_gen2) - self.discriminator(X_real)).mean()

                    # calculate gradient penalty
                    grad_penalty = 0.5 * self.lmbda * self._gradient_penalty(X_real, X_gen1)
                    grad_penalty += 0.5 * self.lmbda * self._gradient_penalty(X_real, X_gen2)

                    # update critic params
                    optim_D.zero_grad()
                    (discr_loss + grad_penalty).backward()
                    optim_D.step()

                    total_grad_penalty += grad_penalty / self.n_critic
                    total_discr_loss += discr_loss.item() / self.n_critic

                #
                # Optimize Generator (n times)
                #
                Z1 = self.Z_dist.sample((real_feat.shape[0], self.z_dim)).to(self.device)
                gen_in1 = torch.cat((Z1, real_feat), dim=1)
                gen_feat1 = self.generator(gen_in1)

                Z2 = self.Z_dist.sample((real_feat.shape[0], self.z_dim)).to(self.device)
                gen_in2 = torch.cat((Z2, real_feat), dim=1)
                gen_feat2 = self.generator(gen_in2)

                X_gen1 = torch.cat((gen_feat1, real_feat), dim=1)
                X_gen2 = torch.cat((gen_feat2, real_feat), dim=1)

                gen_loss = - 0.5 * (torch.mean(self.discriminator(X_gen1)) + torch.mean(self.discriminator(X_gen2)))
                get_tot_loss += gen_loss

                if self.use_cls_loss:
                    classifier = model.cls
                    score = classifier(torch.cat((gen_feat1, gen_feat2), dim=0).view(-1, self.dim, 1, 1))  # N C 1 1
                    target = torch.cat((real_lbl, real_lbl), dim=0).view(-1, 1, 1).long()
                    class_loss = criterion(score, target)
                    get_tot_loss += self.alpha * class_loss

                if self.use_anti_collapse:
                    ar_loss = (1 - F.cosine_similarity(Z1, Z2)) / (1 - F.cosine_similarity(gen_feat1, gen_feat2))
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
            self.generator.load_state_dict(checkpoint['generator'])
            self.generator.to(self.device)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.discriminator.to(self.device)
            if "seed" in checkpoint:
                seed = checkpoint['seed']
                for i in range(len(seed)):
                    self.class_seed[i] = seed[i]

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "seed": self.class_seed,
                 "generator": self.generator.state_dict(), "discriminator": self.discriminator.state_dict()}
        return state


