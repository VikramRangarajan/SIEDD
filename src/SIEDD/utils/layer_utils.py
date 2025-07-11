import torch
import torch.nn as nn
import numpy as np
import math


class Reshape_op(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        assert len(shape) == 3

    def forward(self, x):
        if x.ndim == 3:
            bs, num_features, feat_size = x.shape
        elif x.ndim == 2:
            num_features, feat_size = x.shape
            bs = 1
        else:
            raise ValueError()

        x = x.view(bs, num_features, self.shape[0], self.shape[1], self.shape[2])
        return x


############################
# Initialization scheme
@torch.no_grad()
def hyper_weight_init(m, in_features_main_net, nl=None, siren=False, seed=None):
    nl = "relu" if nl is None else nl
    if hasattr(m, "weight"):
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity=nl, mode="fan_in")
        m.weight.data = m.weight.data / 1e1


@torch.no_grad()
def hyper_weight_init_tensor(m, nl=None, seed=None):
    nl = "relu" if nl is None else nl

    if seed is not None:
        torch.manual_seed(seed)
    nn.init.kaiming_normal_(m, a=0.0, nonlinearity=nl, mode="fan_in")
    m = m / 1e1
    return m


@torch.no_grad()
def hyper_bias_init(m, siren=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    if hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
        m.weight.data = m.weight.data / 1.0e1

    # if hasattr(m, 'bias') and siren:
    #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
    #     with torch.no_grad():
    #         m.bias.uniform_(-1/fan_in, 1/fan_in)


########################
# Initialization methods
@torch.no_grad()
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


@torch.no_grad()
def init_weights_trunc_normal(m):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # if type(m) == BatchLinear or type(m) == nn.Linear:
    if hasattr(m, "weight"):
        fan_in = m.weight.size(1)
        fan_out = m.weight.size(0)
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        mean = 0.0
        # initialize with the same behavior as tf.truncated_normal
        # "The generated values follow a normal distribution with specified mean and
        # standard deviation, except that values whose magnitude is more than 2
        # standard deviations from the mean are dropped and re-picked."
        _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


@torch.no_grad()
def init_weights_linear(m):
    # Default initialization, modified from nn.Linear
    if hasattr(m, "weight"):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    if hasattr(m, "bias") and m.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(m.bias, -bound, bound)


@torch.no_grad()
def init_weights_selu(m):
    if hasattr(m, "weight"):
        num_input = m.weight.size(-1)
        nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


@torch.no_grad()
def init_weights_elu(m):
    # if type(m) == BatchLinear or type(m) == nn.Linear:
    if hasattr(m, "weight"):
        num_input = m.weight.size(-1)
        nn.init.normal_(
            m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input)
        )


@torch.no_grad()
def init_weights_xavier(m):
    # if type(m) == BatchLinear or type(m) == nn.Linear:
    if hasattr(m, "weight"):
        nn.init.xavier_normal_(m.weight)


@torch.no_grad()
def sine_init(omega):
    def _sine_init(m):
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            bound = np.sqrt(6 / num_input) / omega
            # See supplement Sec. 1.5 for discussion of factor 30
            nn.init.uniform_(m.weight, -bound, bound)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.uniform_(m.bias, -bound, bound)

    return _sine_init


@torch.no_grad()
def first_layer_sine_init(m):
    if hasattr(m, "weight"):
        num_input = m.weight.size(-1)
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        bound = 1 / num_input
        nn.init.uniform_(m.weight, -bound, bound)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.uniform_(m.bias, -bound, bound)
