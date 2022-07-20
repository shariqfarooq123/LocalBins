import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        """x.shape NCHW, 
           K.shape N,S,E"""
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1).contiguous(), K.permute(0, 2, 1).contiguous())  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w).contiguous()


class MultiScaleDotProduct(nn.Module):
    def __init__(self):
        super(MultiScaleDotProduct, self).__init__()

    def forward(self, x, K):
        """x.shape NCHW, 
           K.shape N,S,E"""
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"

        N3 = c // 4  # Number of 3x3 kernels. N5 = N3 / 2, N7 = N5/2 and so on
        total = 0
        k_size = 3
        n_channels = N3
        out = []
        while n_channels >= 1:
            kernel = K[:, total : total + n_channels, :]
            y = nn.functional.avg_pool2d(x, k_size, stride=1, padding= k_size //2 )

            y = torch.matmul(y.view(n, c, h*w).permute(0, 2, 1), kernel.permute(0, 2, 1)).permute(0,2,1).view(n, n_channels, h, w)
            out.append(y)

            total += n_channels
            n_channels = n_channels // 2
            k_size += 2

        # rest as 1x1 i.e. without pooling
        kernel = K[:, total : , :]
        y = torch.matmul(x.view(n, c, h*w).permute(0, 2, 1), kernel.permute(0, 2, 1)).permute(0,2,1).view(n, -1, h, w)
        out.append(y)

        y = torch.cat(out, dim=1)
        return y








class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        # blur_kernel=[1, 3, 3, 1],
        # fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / np.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = nn.Linear(style_dim, in_channel)

        self.demodulate = demodulate
        # self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        weight = self.scale * self.weight.squeeze(0)
        style = self.modulation(style)

        if self.demodulate:
            w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
            dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

        input = input * style.reshape(batch, in_channel, 1, 1)
        out = F.conv2d(input, weight, padding=self.padding)

        if self.demodulate:
            out = out * dcoefs.view(batch, -1, 1, 1)

        return out



def positionalencoding2d(d_model, height, width, device='cpu'):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width, device=device)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def relative_positions(seq_len):
    result = []
    for i in range(seq_len):
        front = list(range(-i, 0))
        end = list(range(seq_len - i))
        result.append(front + end)
    return result


import torch
import math
import warnings
from torch import nn, Tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)





class TranslationEmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.learnable_part = nn.Parameter(torch.randn(embedding_dim - 3), requires_grad=True)
        self._embed_net = nn.Sequential(
            nn.Linear(embedding_dim, 2*embedding_dim), 
            nn.GELU(),
            nn.Linear(2*embedding_dim, embedding_dim)
        )

    def forward(self, t):
        b, _three_ = t.shape
        assert _three_ == 3, "Translation vectors should be 3-dimensional"
        lpt = self.learnable_part.unsqueeze(0).expand(b,-1)
        # import pdb; pdb.set_trace()
        x = torch.cat((lpt, t), dim=1)
        return self._embed_net(x)