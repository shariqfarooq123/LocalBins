import torch.nn as nn
from .base import Encoder
import timm

# def build_efficientnet(model_name):

EFFICIENTNET_SETTINGS = {
    "tf_efficientnet_b5_ap": [24, 40, 64, 176, 2048],
    "tf_efficientnetv2_l"  : [32, 64, 96, 224, 1280],
    "tf_efficientnetv2_m"  : [24, 48, 80, 176, 1280],
    "tf_efficientnetv2_s"  : [24, 48, 64, 160, 1280],
    "tf_efficientnet_b0_ap": [16, 24, 40, 112, 1280],

    "mobilenetv2_100"      : [16, 24, 32, 96, 1280]
} 


class EfficientNet(Encoder):
    
    def forward(self, x):
        feats = super().forward(x)
        idx = [4,5,6,8,11]
        return [feats[i] for i in idx]

    @staticmethod
    def build(model_name, pretrained=True):
        basemodel = timm.create_model(model_name, pretrained=pretrained)

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()


        return EfficientNet(basemodel)



