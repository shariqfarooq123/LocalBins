import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import positionalencoding2d
from .miniViT import mViT


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, with_positional_encodings=False):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

        self.pos_enc = None
        self.with_positional_encodings = with_positional_encodings

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        if self.with_positional_encodings:
            if self.pos_enc is None:
                b, c, h, w = concat_with.size()
                self.pos_enc = positionalencoding2d(c, h, w, device=concat_with.device).unsqueeze(0)
            concat_with = concat_with + self.pos_enc
        f = torch.cat([up_x, concat_with], dim=1)

        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048, with_positional_encodings=False):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        self.feature_channels = [num_classes, features // 8, features // 4, features // 2]
        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2, with_positional_encodings=with_positional_encodings)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4,  with_positional_encodings=with_positional_encodings)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8,  with_positional_encodings=with_positional_encodings)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16,  with_positional_encodings=with_positional_encodings)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features, return_multiscale=False):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]


        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        if return_multiscale:
            return [out, x_d3, x_d2, x_d1]
        return out



class MultiScaleDecoderBN(nn.Module):
    def __init__(self, num_features=2048, bottleneck_features=2048):
        super(MultiScaleDecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        self.feature_channels = [features // 16, features // 8, features // 4, features // 2]
        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)


    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return [x_d4, x_d3, x_d2, x_d1]


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features
    

EPS = 1e-3
class Unet(nn.Module):
    def __init__(self, backend, **kwargs):
        super(Unet, self).__init__()
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN(num_classes=1)
    
    def forward(self,x):
        return EPS + torch.relu(self.decoder(self.encoder(x)))

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        # modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        # for m in modules:
        #     yield from m.parameters()
        return self.decoder.parameters()

    @classmethod
    def build(cls, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, **kwargs)
        print('Done.')
        return m

    @staticmethod
    def build_from_config(config):
        return Unet.build(**config)



class UnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear', with_positional_encodings=False, **kwargs):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        self.decoder = DecoderBN(num_classes=128, with_positional_encodings=with_positional_encodings)
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, return_D=False, return_queries=False, return_target_rest=False, return_R=False, return_decoder_features=False, **kwargs):
        unet_out = self.decoder(self.encoder(x), **kwargs)
        D, queries, tgt_rest, bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out, return_D=True, return_queries=True, return_target_rest=True)
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        pred = self.process_Rb(centers, range_attention_maps)
        outputs = []
        if return_D:
            outputs.append(D)
        if return_queries:
            outputs.append(queries)
        if return_target_rest:
            outputs.append(tgt_rest)
        if return_R:
            outputs.append(range_attention_maps)
        if return_decoder_features:
            outputs.append(unet_out)
        outputs.extend([centers, pred])
        return tuple(outputs)

    def process_Rb(self, centers, R):
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1).contiguous()
        out = self.conv_out(R)
        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, min_depth=1e-3, max_depth=10, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, min_val=min_depth, max_val=max_depth, **kwargs)
        print('Done.')
        return m

    @staticmethod
    def build_from_config(config):
        return UnetAdaptiveBins.build(**config)


if __name__ == '__main__':
    model = UnetAdaptiveBins.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)