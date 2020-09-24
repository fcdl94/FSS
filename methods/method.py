import torch
from torch import distributed
from apex.parallel import DistributedDataParallel
from apex import amp


class Method:
    def __init__(self, task, device, logger, opts):
        self.logger = logger
        self.device = device
        self.task = task
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.regularizer = None
        self.model_old = None
        self.novel_classes = self.task.get_n_classes()[-1]

        self.initialize(opts)  # setup the model, optimizer, scheduler and criterion

        if self.model is not None:
            self.model, self.optimizer = amp.initialize(self.model.to(self.device), self.optimizer,
                                                        opt_level=opts.opt_level)
            # Put the model on GPU
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)

    def initialize(self, opts):
        raise NotImplementedError

    def warm_up(self, dataset):
        pass

    def train(self, cur_epoch, train_loader, print_int=10, n_iter=1):
        """Train and return epoch loss"""
        logger = self.logger
        optim = self.optimizer
        scheduler = self.scheduler
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        rloss = torch.tensor([0.]).to(self.device)

        model.train()
        cur_step = 0
        for iteration in range(n_iter):
            for images, labels in train_loader:

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optim.zero_grad()
                outputs = model(images)
                if self.model_old is not None:
                    outputs_old = self.model_old(images)
                    rloss = self.regularizer(outputs, outputs_old)

                # xxx Cross Entropy Loss
                loss = criterion(outputs, labels)  # B x H x W

                loss_tot = loss + rloss

                with amp.scale_loss(loss_tot, optim) as scaled_loss:
                    scaled_loss.backward()

                optim.step()
                scheduler.step()

                epoch_loss += loss.item()
                reg_loss += rloss.item()
                interval_loss += loss_tot.item()

                cur_step += 1
                if cur_step % print_int == 0:
                    interval_loss = interval_loss / print_int
                    logger.info(f"Epoch {cur_epoch}, Batch {cur_step}/{n_iter}*{len(train_loader)},"
                                f" Loss={interval_loss}")
                    logger.debug(f"Loss made of: CE {loss}")
                    # visualization
                    if logger is not None:
                        x = cur_epoch * len(train_loader) + cur_step
                        logger.add_scalar('Loss', interval_loss, x)
                    interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / (len(train_loader) * n_iter)
            reg_loss = reg_loss / distributed.get_world_size() / (len(train_loader) * n_iter)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return epoch_loss, reg_loss

    def validate(self, loader, metrics, ret_samples_ids=None, novel=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        logger = self.logger

        class_loss = 0.0

        ret_samples = []
        with torch.no_grad():
            model.eval()
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)  # B, C, H, W
                if novel:
                    outputs[:, 1:-self.novel_classes] = -float("Inf")

                loss = criterion(outputs, labels)

                class_loss += loss.item()

                _, prediction = outputs.max(dim=1)
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0], prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor([0.]).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)")

        return (class_loss, reg_loss), score, ret_samples

    def load_state_dict(self, checkpoint, strict=True):
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        if strict:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict()}
        return state
