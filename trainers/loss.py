from matplotlib.cbook import flatten
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.cuda.amp as amp

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        with amp.autocast(enabled=False):
            # alpha = 1e-3
            alpha = 0
            g = torch.log(input + alpha) - torch.log(target + alpha)
            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
            
            loss = 10 * torch.sqrt(Dg)
            if not return_interpolated:
                return loss
        return loss, intr_input



class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bin_centers, target_depth_maps):
        # bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        with torch.no_grad():
            target_points = target_depth_maps.flatten(1)  # n, hwc
            mask = target_points.ge(1e-3)  # only valid ground truth points
            target_lengths = mask.float().sum(dim=-1).long()

            # target_points = [p[m] for p, m in zip(target_points, mask)]
            # target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
            # target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

            target_points[~mask] = float('inf')
            target_points, _ = torch.sort(target_points, dim=-1)
            target_points = target_points[...,None]  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss





##################### Multi-Scale Local Bins loss , Pixel wise Chamfer?


from time import time

@torch.jit.script
def compute_target_points(target_depth_maps):
    with torch.no_grad():
            target_points = target_depth_maps.flatten(1)  # n, hwc

            mask = target_points.ge(1e-3)  # only valid ground truth points
            target_points = [p[m] for p, m in zip(target_points, mask)]
            # print(mask.shape)
            target_lengths = mask.sum(dim=1).squeeze().long()
            # print(target_lengths)
            # target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
            target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

    return target_points, target_lengths

class MultiScaleBinsLossV5(nn.Module):
    def __init__(self, config, max_positions=4096):
        super().__init__()
        self.name = "MultiScaleBinsLossV5"
        self.config = config
        self.max_positions = max_positions
        self.wts_on_scale = list(map(float, config.loss_wts_scale.split(",")))
        self.gamma = config.gamma_window
        self.prev_loss = 0

    def get_target(self, bbox_queries, target_depth_maps):
        b, c, h, w = target_depth_maps.shape

        target = {}
        has_at_least_one = False
        for win_size in bbox_queries.window_sizes:
            # target points
            with torch.no_grad():
                # # target points
                # # to get target points, create an list of indices that cover the bboxes and index the depth map
                # # mask out that are invalid
                # # skip if too many are masked or all are masked
                abs_coords = bbox_queries.absolute[win_size]  # B, N, 2; babs[b,n] = [x,y]
                B, N, _two = abs_coords.shape
                k = win_size // 2
                # base_xx, baseyy = np.tile(range(width), height), np.repeat(range(height), width)
                x = torch.arange(-k, k+1)
                y = torch.arange(-k, k+1)
                Y, X = torch.meshgrid(y, x)
                base_coords = torch.stack((X, Y), dim=0)[None, None,...].to(target_depth_maps.device)  # .shape 1, 1, 2, k, k
                
                coords = abs_coords[...,None,None] + base_coords  # shape B, N, 2, k, k
                
                x = coords[:,:,0,:,:]
                y = coords[:,:,1,:,:]
                flatten_indices = y * w + x  # .shape B, N, k, k
                
                flatten_flatten_indices = flatten_indices.flatten(2)  # .shape B, N, kxk
                depths = target_depth_maps.expand(-1,N,-1,-1).flatten(2)  # .shape B, N, HxW
                # print(depths.shape, flatten_flatten_indices.shape, flatten_flatten_indices.max(), flatten_flatten_indices.min())
                
                # target_points = depths[flatten_flatten_indices]  # .shape B, N, kxk
                target_points = torch.gather(depths, dim=-1, index=flatten_flatten_indices.long())

                # merge N boxes into batch
                target_points = target_points.flatten(0,1)  # .shape BxN, kxk


                mask = torch.logical_and(target_points > self.config.min_depth, target_points < self.config.max_depth)

                target_points[~mask] = float('inf')
                target_points = torch.sort(target_points, dim=-1).values
                target_lengths = torch.sum(mask.long(), dim=-1)

                # find out which of the boxes have mostly invalid regions >50%
                portion = target_lengths.float() / float(target_points.size(1))
                valids = portion > 0.5
                target[win_size] = target_points, target_lengths, valids

                if valids.sum() > 0:
                    has_at_least_one = True

        if not has_at_least_one:
            return None
        return target
        


    def forward(self, response, bbox_queries, target_depth_maps):
        """
        bbox_queries : utils.bbox_utils.RandomBBoxQueries; .absolute = dict[win_size:int -> x_y:Tensor(B,N,2) :int]
        response : {scale: {win_size : centers - Tensor(B, N, n_bins)}}
        bin_centers  : dict[win_size:int -> bin_centers:Tensor(B,N,n_bins) :float]
        target_depth_maps : B,1,H,W
        """
        b, c, h, w = target_depth_maps.shape

        total = 0
        window_sizes = bbox_queries.window_sizes
        target = self.get_target(bbox_queries, target_depth_maps)
        if target is None:
            # no valid bbox across all window sizes
            # print("Woaah target is None")
            return None  # wtf?

        # order response according to scale factor in descending order to bring to same order as in wts
        for s_wt, (s, bin_centers) in zip(self.wts_on_scale, sorted(response.items(), key=lambda x: x[0], reverse=True)):
            if s_wt <= 0:
                continue

            total_at_this_scale = 0
            valid_sizes = len(window_sizes)
            wc = 1.0  # initial loss weight for smallest window
            sum_w = 0
            for win_size in sorted(window_sizes):
                # input points 
                input_points = bin_centers[win_size]  # .shape B, N, n_bins
                n = input_points.shape[1]
                # merge N boxes with batch
                input_points = input_points.flatten(0,1)  # .shape BxN, n_bins

                # target points
                abs_coords = bbox_queries.absolute[win_size]  # B, N, 2; babs[b,n] = [x,y]
                # print(abs_coords.shape)
                # xc, yc = abs_coords[...,0], abs_coords[...,1]
                B, N, _two = abs_coords.shape
                assert B == b, f"Batch sizes should match, got {b} whereas queries have batch size {B}"
                assert n == N, f"Number of bboxes should match, got {n} and {N}"

                target_points, target_lengths, valids = target[win_size]
                # print(valids.sum(), input_points.shape, target_points.shape, target_lengths.shape, portion.shape)

                input_points, target_points, target_lengths = input_points[valids], target_points[valids], target_lengths[valids]
                if len(input_points) == 0: # "Could not find even one valid BBOX for this window size"
                    valid_sizes -= 1
                    wc = self.gamma * wc  # still reduce the weight by factor of gamma
                    continue

                loss, _ = chamfer_distance(x=input_points[...,None], y=target_points[...,None], y_lengths=target_lengths)
                total_at_this_scale = total_at_this_scale + wc * loss
                sum_w += wc
                wc = self.gamma * wc  # reduce weight
            
            if valid_sizes <= 0:  # "Could not find even one valid BBOX across any window size"
                # should never reach here
                print("##################################\n\nWooah!! should never reach here!!\n\n##################################")
                return None 

            total = total + s_wt * (total_at_this_scale / sum_w)
        return total

