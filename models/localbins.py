"""
previous: v2: Instead of prediction of local bins at every pixel, model receives bbox queries
bbox query = (xc, yc, scale)

v3: ViT for global!!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import positionalencoding2d
from .miniViT import mViT
from .backbone import EfficientNet, EFFICIENTNET_SETTINGS
from .backbone import DenseNet161, DENSENET_SETTINGS

import numpy as np
from torchvision.ops import roi_align as torch_roi_align

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


class SeedBinRegressor(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):  
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0), 
            nn.GELU(),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.ReLU()
        )

    def forward(self,x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B = self._net(x)
        eps = 1e-3
        B = B + eps
        B_widths_normed = B / B.sum(dim=1, keepdim=True)
        B_widths = (self.max_depth - self.min_depth) * B_widths_normed  # .shape NCHW
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(B_widths, (0,0,0,0,0,1), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        return B_widths_normed, B_centers

class Projector(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0),
        )

    def forward(self, x):
        return self._net(x)

class LinearSplitter(nn.Module):
    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=1e-3, max_depth=10):
        super().__init__()

        self.prev_nbins = prev_nbins
        self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mlp_dim, prev_nbins * split_factor, 1, 1, 0),
            nn.ReLU()
        )
    
    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding
        S = self._net(x)
        eps = 1e-3
        S = S + eps
        n, c, h, w = S.shape
        S = S.view(n, self.prev_nbins, self.split_factor, h, w)
        S_normed = S / S.sum(dim=2, keepdim=True)  # fractional splits

        b_prev = nn.functional.interpolate(b_prev, (h,w), mode='bilinear', align_corners=True)
        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)  # renormalize for gurantees
        # print(b_prev.shape, S_normed.shape)
        # if is_for_query:(1).expand(-1, b_prev.size(0)//n, -1, -1, -1, -1).flatten(0,1)  # TODO ? can replace all this with a single torch.repeat?
        b = b_prev.unsqueeze(2) * S_normed
        b = b.flatten(1,2)  # .shape n, prev_nbins * split_factor, h, w

        # calculate bin centers for loss calculation
        B_widths = (self.max_depth - self.min_depth) * b  # .shape N, nprev * splitfactor, H, W
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(B_widths, (0,0,0,0,0,1), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        return b, B_centers

class SigmoidSplitter(nn.Module):
    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=1e-3, max_depth=10):
        super().__init__()

        self.prev_nbins = prev_nbins
        self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mlp_dim, prev_nbins, 1, 1, 0),
            # nn.ReLU()
        )
    
    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding
        S = torch.sigmoid(self._net(x))  # .shape n,c, h, w; 0<S<1
        n,c, h, w = S.shape
        S = S.unsqueeze(2)  # .shape n, c, 1, h, w
        S_normed = torch.cat((S, 1-S), dim=2)  # fractional splits , .shape n, prev_nbins, 2, h, w
        # eps = 1e-3
        # S = S + eps
        # n, c, h, w = S.shape
        # S = S.view(n, self.prev_nbins, self.split_factor, h, w)
        # S_normed = S / S.sum(dim=2, keepdim=True)  # fractional splits

        b_prev = nn.functional.interpolate(b_prev, (h,w), mode='bilinear', align_corners=True)
        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)  # renormalize for gurantees
        # print(b_prev.shape, S_normed.shape)
        # if is_for_query:(1).expand(-1, b_prev.size(0)//n, -1, -1, -1, -1).flatten(0,1)  # TODO ? can replace all this with a single torch.repeat?
        b = b_prev.unsqueeze(2) * S_normed
        b = b.flatten(1,2)  # .shape n, prev_nbins * split_factor, h, w

        # calculate bin centers for loss calculation
        B_widths = (self.max_depth - self.min_depth) * b  # .shape N, nprev * splitfactor, H, W
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(B_widths, (0,0,0,0,0,1), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        return b, B_centers



class NaiveSplitter(nn.Module):
    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=1e-3, max_depth=10):
        super().__init__()

        self.prev_nbins = prev_nbins
        self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.register_buffer('alpha', 0.5*torch.ones(1,1,1,1,1, dtype=float))

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding

        n, c, h, w = x.shape
        S_normed = self.alpha.expand(n, self.prev_nbins, 2, h, w)

        b_prev = nn.functional.interpolate(b_prev, (h,w), mode='bilinear', align_corners=True)
        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)  # renormalize for gurantees
        # print(b_prev.shape, S_normed.shape)
        # if is_for_query:(1).expand(-1, b_prev.size(0)//n, -1, -1, -1, -1).flatten(0,1)  # TODO ? can replace all this with a single torch.repeat?
        b = b_prev.unsqueeze(2) * S_normed
        b = b.flatten(1,2)  # .shape n, prev_nbins * split_factor, h, w

        # calculate bin centers for loss calculation
        B_widths = (self.max_depth - self.min_depth) * b  # .shape N, nprev * splitfactor, H, W
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(B_widths, (0,0,0,0,0,1), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        return b, B_centers



class IdentitySplitter(nn.Module):
    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=1e-3, max_depth=10):
        super().__init__()

        # self.prev_nbins = prev_nbins
        # self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth
        # self.register_buffer('alpha', 0.5*torch.ones(1,1,1,1,1, dtype=float))

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding

        n, c, h, w = x.shape
        b_prev = nn.functional.interpolate(b_prev, (h,w), mode='bilinear', align_corners=True)
        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)  # renormalize for gurantees
        b = b_prev
        # calculate bin centers for loss calculation
        B_widths = (self.max_depth - self.min_depth) * b  # .shape N, nprev * splitfactor, H, W
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(B_widths, (0,0,0,0,0,1), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        return b, B_centers





class DecoderBN(nn.Module):
    def __init__(self, in_channels=EFFICIENTNET_SETTINGS["tf_efficientnet_b5_ap"], n_seed_bins=8, split_factor=2,
     out_channels=128, bin_embedding_dim=128, with_positional_encodings=False, min_depth=1e-3, max_depth=10, splitter_type='linear', **kwargs):
        super(DecoderBN, self).__init__()

        # features = int(num_features)
        bottleneck_features = features = in_channels[-1]
        self.n_seed_bins = n_seed_bins
        self.split_factor = split_factor
        self.bin_embedding_dim = bin_embedding_dim

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        num_in_features = [features // 1 + in_channels[-2], features // 2 + in_channels[-3], features // 4 + in_channels[-4], features // 8 + in_channels[-5]]
        num_out_features = [features // 2,           features // 4,           features // 8,           features // 16]
        self.scales = [16, 8, 4, 2]  # spatial scale factors
        n_total_bins = n_seed_bins * (split_factor ** len(num_out_features))

        self.up_blocks = nn.ModuleList([
                                    UpSampleBN(num_in, num_out, with_positional_encodings) 
                                    for num_in, num_out in zip(num_in_features, num_out_features)
                                ])

        self.seed_bin_regressor = SeedBinRegressor(features, n_bins=n_seed_bins if not (splitter_type == 'identity') else n_total_bins , min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(features, bin_embedding_dim)  # TODO: sedd bin regrssor on projector

        self.projectors = nn.ModuleList([
                                    Projector(num_out, bin_embedding_dim)
                                    for num_out in num_out_features
                                ])

        type2splitter = dict(linear=LinearSplitter, naive=NaiveSplitter, sigmoid = SigmoidSplitter, identity=IdentitySplitter)
        Splitter =type2splitter[splitter_type]
        self.splitters = nn.ModuleList([
                                    Splitter(bin_embedding_dim, n_seed_bins * (split_factor**i), split_factor, min_depth=min_depth, max_depth=max_depth)
                                    for i in range(len(num_out_features))
                                ])
        
        self.height_head = nn.Sequential(
            nn.Conv2d(features // 16, 256, kernel_size=1, stride=1, padding=0), 
            nn.GELU(),
            nn.Conv2d(256,256,1,1,0),
            nn.GELU(),
            nn.Conv2d(256,1, 1, 1, 0)
        )  # NOTE: NOT USED IN FINAL VERSION
        # self.conv_out = nn.Conv2d(features // 16, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(features // 16, n_seed_bins * (split_factor ** len(num_out_features)), kernel_size=3, stride=1, padding=1)

    def forward(self, features, queries=None, roi_align_mode=0, t=1., return_final_centers=False):
        """
        features : List of features from encoder 
        queries : dict of queries, {win_size: q[B,N,2]]}
        """

        x_block4 = features[-1]
        x_blocks = features[:-1][::-1]

        x_d0 = self.conv2(x_block4)
        x = x_d0
        seed_b, seed_b_centers = self.seed_bin_regressor(x)
        b_prev = seed_b.clone()
        prev_b_embedding = self.seed_projector(x)

        if queries is not None:
            response_dict = {}
            prev_data = {}
            for win_size, q in queries.items():
                prev_data[win_size] = {}
                pooled_bttlnck = self.roi_align(x, q, win_size, scale=32, roi_align_mode=roi_align_mode)
                # seed_b.shape = (b, n_seed_bins, H/32, W/32)
                # we need to expand for num boxes
                num_boxes = q.size(1)
                q_seed_b, q_seed_b_centers = self.seed_bin_regressor(pooled_bttlnck)
                prev_data[win_size]['b_prev']  = q_seed_b
                prev_data[win_size]['prev_b_embedding'] = self.seed_projector(pooled_bttlnck)
        else:
            response_dict = None
            

        for up_block, projector, splitter, scale, x_block in zip(self.up_blocks, self.projectors, self.splitters, self.scales, x_blocks):
            x = up_block(x, x_block)
            b_embedding = projector(x)
            b, b_centers = splitter(b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

            if queries is not None:
                out_dict, prev_data = self.handle_queries(x, queries, projector, splitter, scale, prev_data, roi_align_mode=roi_align_mode)
                response_dict[scale] = out_dict

        pred_height = self.height_head(x)
        x = self.conv_out(x)
        

        x = torch.softmax(x/t, dim=1)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        out = torch.sum(x * b_centers, dim=1, keepdim=True)
        if return_final_centers:
            return response_dict, pred_height, b_centers, out
        return response_dict, pred_height, out

    def roi_align(self, feature_block, bboxes, win_size, scale, roi_align_mode=0):
        b, c, h, w = feature_block.shape

        k  = win_size // 2
        batch_size, num_boxes, _two = bboxes.shape

        if roi_align_mode < 0:
            # use pytorch roi_align
            # convert bboxes to shape K, 5 : i, xmin, ymin, xmax, ymax
            t_boxes = torch.cat((bboxes - k, bboxes + k), dim=-1).view(-1,4)  # torch boxes: b* n , 4
            inds = torch.arange(batch_size).repeat_interleave(num_boxes)[:,None].to(feature_block.device)  # .shape b*n, 1
            t_boxes = torch.cat((inds, t_boxes), dim=-1)  # .shape b*n, 5

            pooled = torch_roi_align(feature_block, t_boxes, (1,1), 1/scale)  # .shape b*n, c, 1, 1
            return pooled


        # base_xx, baseyy = np.tile(range(width), height), np.repeat(range(height), width)
        # sampling_ratio = 
        x = torch.linspace(-k, k, steps=k, device=feature_block.device)
        y = torch.linspace(-k, k, steps=k, device=feature_block.device)
        Y, X = torch.meshgrid(y, x)

        base_coords = torch.stack((X, Y), dim=0)[None, None,...]  # .shape 1, 1, 2, k, k
        
        uv = bboxes[...,None,None] + base_coords  # shape B, N, 2, k, k
        b, n, *_ = uv.shape

        if roi_align_mode == 0:
            # uv.shape = B, N, 2, k, k

            # change to [-1,1]
            H, W = scale * h, scale * w 
            size = torch.tensor((H - 1, W - 1), dtype=uv.dtype).view(1, 1, -1, 1, 1).to(x.device)
            grid = ((2 * uv / size - 1)).permute(0, 1, 3, 4, 2)  # B, N, k, k, 2

            # print(b, batch_size, num_boxes)
            # feature_block.shape = batch_size, c, h ,w
            pooled_features = []
            # chunk_size = 5
            for i in range(0, num_boxes):
                # print(feature_block.shape)
                # f = feature_block.unsqueeze(1).expand(-1, chunk_size, -1, -1, -1).flatten(0,1)  # .shape B * N, C, h, w
                # g = grid[:,i:i+chunk_size,...]
                # m = g.size(1)  # = chunk_size for all except the last one (for which its equal to remaining)
                # interpolated = nn.functional.grid_sample(f, g.reshape(batch_size * m, k, k, 2), align_corners=False)  # .shape B * N, C, k, k 
                interpolated = nn.functional.grid_sample(feature_block, grid[:,i,...], align_corners=False)  # .shape B * 1, C, k, k 
                pooled = nn.functional.adaptive_avg_pool2d(interpolated, (1,1)).squeeze(1) # .shape B, 1 ,C, 1, 1
                pooled_features.append(pooled)

            pooled = torch.cat(pooled_features, dim=1).view(batch_size * num_boxes, c, 1, 1)  # B*N, C, 1, 1
            return pooled

        uv = uv.flatten(0,1) # shape B * N, 2, k, k
        

        # change to [-1,1]
        H, W = scale * h, scale * w 
        size = torch.tensor((H - 1, W - 1), dtype=uv.dtype).view(1, -1, 1, 1).to(x.device)
        grid = ((2 * uv / size - 1)).permute(0,2,3,1)  # B * N, k, k, 2

        # print(b, batch_size, num_boxes)
        feature_block = feature_block.unsqueeze(1).expand(-1, num_boxes, -1, -1, -1).flatten(0,1)  # .shape B * N, C, h, w
        interpolated = nn.functional.grid_sample(feature_block, grid, align_corners=False)  # .shape B * N, C, k, k 
        pooled = nn.functional.adaptive_avg_pool2d(interpolated, (1,1))  # .shape B * N, C, 1, 1
        return pooled

        

    def handle_queries(self, feature_block, queries, projector, splitter, scale, prev_data, roi_align_mode=0):
        """
        feature_block : B, C, H, W
        queries = dict win_size : B, N, 2 ; (xc,yc)
        response_dict : dict[scale][win_size] : B, N, n_bins
        """
        batch_size, c, h, w = feature_block.shape
        out_dict = {}
        curr_data = {}
        for win_size, q in queries.items():
            pooled = self.roi_align(feature_block, q, win_size, scale, roi_align_mode=roi_align_mode)  # .shape b * q.size(1), c, 1, 1
            b_emb = projector(pooled)
            b, b_centers = splitter(b_emb, prev_data[win_size]['b_prev'], prev_data[win_size]['prev_b_embedding'], interpolate=False, is_for_query=True)
            out_dict[win_size] = b_centers.view(batch_size, q.size(1), b_centers.size(1))

            curr_data[win_size] = {}
            curr_data[win_size]['b_prev'] = b
            curr_data[win_size]['prev_b_embedding'] = b_emb


        return out_dict, curr_data


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
    

def build_encoder(backbone):
    if "dense" in backbone:
        return DenseNet161(), DENSENET_SETTINGS[backbone]
    else:
        return EfficientNet.build(backbone), EFFICIENTNET_SETTINGS[backbone]


EPS = 1e-3
class UnetLocalBins(nn.Module):
    def __init__(self, backbone, n_bins=256, min_depth=1e-3, max_depth=10, roi_align_mode=-1, **kwargs):
        super(UnetLocalBins, self).__init__()

        num_decoder_out = 128
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.roi_align_mode = roi_align_mode

        # self.encoder = EfficientNet.build(backbone)
        model, settings = build_encoder(backbone)
        self.encoder = model
        self.decoder = DecoderBN(in_channels=settings, n_local_bins=256, out_channels=num_decoder_out, min_depth=min_depth, max_depth=max_depth, **kwargs)
        # self.window_sizes = self.decoder.window_sizes



        # self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
        #                         nn.Softmax(dim=1))

    def forward(self,x, queries=None, **kwargs):
        return self.decoder(self.encoder(x), queries, roi_align_mode=self.roi_align_mode, **kwargs)


    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        # modules = [self.decoder, self.adaptive_bins_layer, self.conv_out, self.height_head]
        # for m in modules:
        #     yield from m.parameters()
        return self.decoder.parameters()

    @classmethod
    def build(cls, backbone='tf_efficientnet_b5_ap', **kwargs):
        return cls(backbone, **kwargs)

    @staticmethod
    def build_from_config(config):
        return UnetLocalBins.build(**config)




