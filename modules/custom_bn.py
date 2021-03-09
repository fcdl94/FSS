import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN, InPlaceABNSync, functions as in_funcs
import torch.distributed as distributed
from inplace_abn import _backend

class AIN(nn.Module):
    """Activated Instance Normalization

    This gathers a InstanceNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    activation : str
        Name of the activation functions, one of: `relu`, `leaky_relu`, `elu` or `identity`.
    activation_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 activation="leaky_relu", activation_param=0.01, group=distributed.group.WORLD):
        super(AIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.activation = activation
        self.activation_param = activation_param

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

        self.group = group

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.instance_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                     self.training or not self.track_running_stats, self.momentum, self.eps)

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Post-Pytorch 1.0 models using standard BatchNorm have a "num_batches_tracked" parameter that we need to ignore
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        if not self.track_running_stats:
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    state_dict.pop(key)

        super(AIN, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                               error_msgs, unexpected_keys)

    def extra_repr(self):
        rep = '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
        if self.activation in ["leaky_relu", "elu"]:
            rep += '[{activation_param}]'
        return rep.format(**self.__dict__)


class ABR(nn.Module):
    """Activated Batch Renormalization

    This gathers a BatchNorm and an activation function in a single module
    Adapted from https://arxiv.org/pdf/1702.03275.pdf

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    activation : str
        Name of the activation functions, one of: `relu`, `leaky_relu`, `elu` or `identity`.
    activation_param : float
        Negative slope for the `leaky_relu` activation.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.0, affine=True, activation="leaky_relu",
                 activation_param=0.01, group=distributed.group.WORLD, renorm=True):
        super(ABR, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.activation_param = activation_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

        self.group = group

        self.renorm = renorm
        self.momentum = momentum

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        if not self.renorm or not self.training:  # if eval, don't renorm
            weight = self.weight
            bias = self.bias
        else:
            with torch.no_grad():
                running_std = (self.running_var + self.eps).pow(0.5)
                xt = x.transpose(1, 0).reshape(x.shape[1], -1)
                r = (xt.var(dim=1) + self.eps).pow(0.5) / running_std
                d = (xt.mean(dim=1) - self.running_mean) / running_std
            weight = self.weight * r
            bias = self.bias + self.weight * d

        x = functional.batch_norm(x, self.running_mean, self.running_var, weight, bias,
                                  self.training, self.momentum, self.eps)

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Post-Pytorch 1.0 models using standard BatchNorm have a "num_batches_tracked" parameter that we need to ignore
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(ABR, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                               error_msgs, unexpected_keys)

    def extra_repr(self):
        rep = '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
        if self.activation in ["leaky_relu", "elu"]:
            rep += '[{activation_param}]'
        return rep.format(**self.__dict__)


class InPlaceABR(ABR):
    def __init__(self, num_features, eps=1e-5, momentum=0.0, affine=True, activation="leaky_relu",
                 activation_param=0.01):
        super().__init__(num_features, eps, momentum, affine, activation, activation_param)

    def forward(self, x):
        if not self.renorm or not self.training:  # if eval, don't renorm
            weight = self.weight
            bias = self.bias
        else:
            with torch.no_grad():
                running_std = (self.running_var + self.eps).pow(0.5)
                xt = x.transpose(1, 0).reshape(x.shape[1], -1)
                r = (xt.var(dim=1) + self.eps).pow(0.5) / running_std
                d = (xt.mean(dim=1) - self.running_mean) / running_std
            weight = self.weight * r
            bias = self.bias + self.weight * d

        x = in_funcs.inplace_abn(
            x, weight, bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps,
            self.activation, self.activation_param)

        if isinstance(x, tuple) or isinstance(x, list):  # to be compatible with inplace version < 1.0
            x = x[0]

        return x


class InPlaceABR_R(ABR):
    def __init__(self, num_features, eps=1e-5, momentum=0.0, affine=True, activation="leaky_relu",
                 activation_param=0.01):
        super().__init__(num_features, eps, momentum, affine, activation, activation_param)
        self.alpha = 0.9

    def forward(self, x):
        if not self.renorm or not self.training:  # if eval, don't renorm
            weight = self.weight
            bias = self.bias
        else:
            with torch.no_grad():
                running_std = (self.running_var + self.eps).pow(0.5)
                xt = x.transpose(1, 0).reshape(x.shape[1], -1)
                new_std = (1-self.alpha) * xt.var(dim=1) + self.alpha * running_std
                r = (xt.var(dim=1) + self.eps).pow(0.5) / new_std
                d = self.alpha * (xt.mean(dim=1) - self.running_mean) / new_std
            weight = self.weight * r
            bias = self.bias + self.weight * d

        x = in_funcs.inplace_abn(
            x, weight, bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps,
            self.activation, self.activation_param)

        if isinstance(x, tuple) or isinstance(x, list):  # to be compatible with inplace version < 1.0
            x = x[0]

        return x


class RandABIN(ABN):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
                 activation_param=0.01):
        super().__init__(num_features, eps, momentum, affine, activation, activation_param)
        self.old_running_mean = None
        self.old_running_var = None

    def forward(self, x):
        if self.training:
            if self.old_running_mean is None:
                self.old_running_mean = self.running_mean.clone()
                self.old_running_var = self.running_var.clone()
            shape = [*x.shape]
            # shape[0] = 10  # 24 - shape[0]  # make total batch size = 24
            xs = (torch.randn(shape[:2], device=x.device, requires_grad=False))
            xs *= self.old_running_var.pow(0.5).view(1, self.num_features)
            xs += self.old_running_mean.view(1, self.num_features)
            xs = torch.ones(shape, device=x.device, requires_grad=False) * xs.view(*xs.shape, 1, 1)
            xs = torch.cat((x, xs), dim=0)
        else:
            xs = x
        xs = functional.batch_norm(xs, self.running_mean, self.running_var, self.weight, self.bias,
                                   self.training, self.momentum, self.eps)

        x = xs[:len(x)] if self.training else xs  # if training, remove noisy samples

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))


class RandABN(ABN):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
                 activation_param=0.01):
        super().__init__(num_features, eps, momentum, affine, activation, activation_param)
        self.old_running_mean = None
        self.old_running_var = None

    def forward(self, x):
        if self.training:
            if self.old_running_mean is None:
                self.old_running_mean = self.running_mean.clone()
                self.old_running_var = self.running_var.clone()
            shape = [*x.shape]
            # shape[0] = 10  # 24 - shape[0]  # make total batch size = 24
            xs = (torch.randn(shape[:2], device=x.device, requires_grad=False))
            xs *= self.old_running_var.pow(0.5).view(1, self.num_features)
            xs += self.old_running_mean.view(1, self.num_features)
            xs = torch.ones(shape, device=x.device, requires_grad=False) * xs.view(*xs.shape, 1, 1)
            xs = torch.cat((x, xs), dim=0)
        else:
            xs = x
        xs = functional.batch_norm(xs, self.running_mean, self.running_var, self.weight, self.bias,
                                   self.training, self.momentum, self.eps)

        x = xs[:len(x)] if self.training else xs  # if training, remove noisy samples

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))


class RandInPlaceABNSync(InPlaceABNSync):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
                 activation_param=0.01, group=distributed.group.WORLD):
        super().__init__(num_features, eps, momentum, affine, activation, activation_param)
        self.group = group
        self.old_running_mean = None
        self.old_running_var = None

    def forward(self, x):
        if self.training:
            if self.old_running_mean is None:
                self.old_running_mean = self.running_mean.clone()
                self.old_running_var = self.running_var.clone()
            shape = [*x.shape]
            # shape[0] = 10  # 24 - shape[0]  # make total batch size = 24
            xs = (torch.randn(shape[:2], device=x.device, requires_grad=False))
            xs *= self.old_running_var.pow(0.5).view(1, self.num_features)
            xs += self.old_running_mean.view(1, self.num_features)
            xs = torch.ones(shape, device=x.device, requires_grad=False) * xs.view(*xs.shape, 1, 1)
            xs = torch.cat((x, xs), dim=0)
        else:
            xs = x

        xs, _, _ = in_funcs.inplace_abn_sync(
            xs, self.weight, self.bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps,
            self.activation, self.activation_param, self.group)

        x = xs[:len(x)] if self.training else xs  # if training, remove noisy samples

        return x
