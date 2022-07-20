import base64
import math
import re
from io import BytesIO

import matplotlib.cm
import numpy as np
import torch
import torch.nn
from PIL import Image


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data.distributed
import wandb
import os, subprocess


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()
        
        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


# def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
#     value = value.cpu().numpy()[0, :, :]
#     invalid_mask = value == -1

#     # normalize
#     vmin = value.min() if vmin is None else vmin
#     vmax = value.max() if vmax is None else vmax
#     if vmin != vmax:
#         value = (value - vmin) / (vmax - vmin)  # vmin..vmax
#     else:
#         # Avoid 0-division
#         value = value * 0.
#     # squeeze last dim if it exists
#     # value = value.squeeze(axis=0)
#     cmapper = matplotlib.cm.get_cmap(cmap)
#     value = cmapper(value, bytes=True)  # (nxmx4)
#     value[invalid_mask] = 255
#     img = value[:, :, :3]

#     #     return img.transpose((2, 0, 1))
#     return img



import matplotlib


def colorize(value, vmin=10, vmax=1000, cmap='magma_r', invalid_val = -99, invalid_mask = None, background_color=(128,128,128,1)):
    """
    value : a depth map
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()


    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = value[mask].min() if vmin is None else vmin
    vmax = value[mask].max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    # white out the invalid values
    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    return img


def log_images(img, depth, pred, config, step):
    depth = colorize(depth, vmin=config.min_depth, vmax=config.max_depth)
    pred = colorize(pred, vmin=config.min_depth, vmax=config.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_metrics(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        # do_kb_crop = config.do_kb_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval

    if gt.shape[-1] != pred.shape[-1] and interpolate:
        pred = nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)
   


    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
            int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask[45:471, 41:601] = 1
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask])





#################################### Model uilts ################################################
def parallelize(config, model, find_unused_parameters=True):
    # config. configtraining']
    # config.gpu = gpu

    if config.gpu is not None: 
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)

    config.multigpu = False
    if config.distributed:
        # Use DDP
        config.multigpu = True
        config.rank = config.rank * config.ngpus_per_node + config.gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        config.batch_size = int(config.batch_size / config.ngpus_per_node)
        # config.batch_size = 8
        config.workers = int((config.num_workers + config.ngpus_per_node - 1) / config.ngpus_per_node)
        print(config.gpu, config.rank, config.batch_size, config.workers)
        torch.cuda.set_device(config.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(config.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], output_device=config.gpu,
                                                          find_unused_parameters=find_unused_parameters)

    elif config.gpu is None:
        # Use DP
        config.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model

def get_adabins(pretrained=True):
    from models.unet_adaptive_bins import UnetAdaptiveBins

    adabins = UnetAdaptiveBins.build(256)
    if pretrained:
        adabins.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/shariqfarooq123/AdaBins/releases/download/v1.0/AdaBins_nyu-256-2fb686a.pth"))
    
    adabins.eval()
    
    return adabins


#################################################################################################


##################################### Demo Utilities ############################################
def b64_to_pil(b64string):
    image_data = re.sub('^data:image/.+;base64,', '', b64string)
    # image = Image.open(cStringIO.StringIO(image_data))
    return Image.open(BytesIO(base64.b64decode(image_data)))


# Compute edge magnitudes
from scipy import ndimage


def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)


class PointCloudHelper():
    def __init__(self, width=640, height=480):
        self.xx, self.yy = self.worldCoords(width, height)

    def worldCoords(self, width=640, height=480):
        hfov_degrees, vfov_degrees = 57, 43
        hFov = math.radians(hfov_degrees)
        vFov = math.radians(vfov_degrees)
        cx, cy = width / 2, height / 2
        fx = width / (2 * math.tan(hFov / 2))
        fy = height / (2 * math.tan(vFov / 2))
        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        return xx, yy

    def depth_to_points(self, depth):
        depth[edges(depth) > 0.3] = np.nan  # Hide depth edges
        length = depth.shape[0] * depth.shape[1]
        # depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
        z = depth.reshape(length)

        return np.dstack((self.xx * z, self.yy * z, z)).reshape((length, 3))

#####################################################################################################
import glob
def find_ckpt(pattern, checkpoint_dir='./checkpoints', ckpt_type='best'):
    matches = glob.glob(os.path.join(checkpoint_dir,f"*{pattern}*{ckpt_type}*"))
    if not (len(matches) > 0):
        raise ValueError(f"No matches found for the pattern {pattern}")

    checkpoint = matches[0]
    return checkpoint

################# wandb utils
from easydict import EasyDict as edict
def fetch_model_config(pattern, project=os.environ.get("WANDB_PROJECT"), **overwrite_kwargs):
    api = wandb.Api()
    query = None
    runs = api.runs(project,filters=query, order="+summary_metrics.Metrics/abs_rel")


    for r in runs:
        if not pattern in r.name:
            continue

        model_name = r.config["model"]
        version_name =r.config["version_name"]

        conf = {**r.config, **overwrite_kwargs}
        return edict(conf)
