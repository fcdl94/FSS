import utils
import torch
import torch.nn.functional as F
import torch.nn as nn

class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()


def get_scheduler(opts, optim):
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optim, max_iters=opts.max_iter, power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=opts.lr_decay_step,
                                                    gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError
    return scheduler


def get_batch(it, dataloader):
    try:
        batch = next(it)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        it = iter(dataloader)
        batch = next(it)
    return it, batch


def get_prototype(model, ds, cl, device, interpolate_label=True, return_all=False, background=False):
    protos = []
    bkg_proto = []
    for img, lbl in ds:
        img, lbl = img.to(device), lbl.to(device)
        out = model(img.unsqueeze(0), use_classifier=False)
        if interpolate_label:  # to match output size
            lbl = F.interpolate(lbl.float().view(1, 1, lbl.shape[0], lbl.shape[1]),
                                size=out.shape[-2:], mode="nearest").view(out.shape[-2:]).type(torch.uint8)
        else:  # interpolate output to match label size
            out = F.interpolate(out, size=img.shape[-2:], mode="bilinear", align_corners=False)
        out = out.squeeze(0)
        out = out.view(out.shape[0], -1).t()  # (HxW) x F
        lbl = lbl.flatten()  # Now it is (HxW)
        if (lbl == cl).float().sum() > 0 and (lbl != cl).float().sum() > 0:
            protos.append(norm_mean(out[lbl == cl, :]))
            bkg_proto.append(norm_mean(out[lbl != cl, :]))

    if len(protos) > 0:
        protos = torch.cat(protos, dim=0)
        bkg_proto = torch.cat(bkg_proto, dim=0)

        if return_all:
            return protos if not background else (protos, bkg_proto)

        return protos.mean(dim=0) if not background else (protos.mean(dim=0), bkg_proto.mean(dim=0))
    else:
        return None


def norm_mean(x):
    # x should be N x F, return 1 x F
    return F.normalize(x, dim=1).mean(dim=0, keepdim=True)


class myReLU(torch.nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inchannels=None, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
