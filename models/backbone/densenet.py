import torch
import torch.nn as nn
import timm
from .base import Encoder

DENSENET_SETTINGS = {
    "densenet161": [96, 384, 768, 2112, 2208]
}

class DenseNet161(nn.Module):
    def __init__(self):
        super(DenseNet161, self).__init__()
        self.net = timm.create_model("densenet161", pretrained=True)
        self.net.global_pool = nn.Identity()
        self.net.classifier = nn.Identity()
        self._handles = []
        self._features = {}
        self.create_hook("conv0", 0)
        self.create_hook("denseblock1", 1)
        self.create_hook("denseblock2", 2)
        self.create_hook("denseblock3", 3)
        self.create_hook("norm5", 4)

    def create_hook(self, layername, activation_name):
        def hook(module, input, output):
            self._features[activation_name] = output
            return output
        handle = getattr(self.net.features, layername).register_forward_hook(hook)
        self._handles.append(handle)

    def remove_handles(self):
        for handle in self._handles:
            handle.remove()

    def forward(self,x):
        self.net(x)
        return [self._features[i] for i in range(5)]
