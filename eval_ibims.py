import torch
from torch.utils.data import Dataset, DataLoader
from utils.misc import compute_metrics, RunningAverageDict
from utils.config import get_config, DATASETS_CONFIG
from utils.model_io import load_wts
import torch
import numpy as np
from tqdm import tqdm
import argparse
from models.builder import build_model
from utils.arg_utils import parse_unknown
import glob
import os
from pprint import pprint
from PIL import Image
from torchvision import transforms as T
import wandb

class iBims(Dataset):
    def __init__(self, root_folder=r'../ibims/ibims1_core_raw/'):
        with open(os.path.join(root_folder, "imagelist.txt"), 'r') as f:
            imglist = f.read().split()

        samples = []
        for basename in imglist:
            img_path = os.path.join(root_folder, 'rgb', basename + ".png")
            depth_path = os.path.join(root_folder, 'depth', basename + ".png")
            valid_mask_path = os.path.join(root_folder, 'mask_invalid', basename+".png")
            transp_mask_path = os.path.join(root_folder, 'mask_transp', basename+".png")

            samples.append((img_path, depth_path, valid_mask_path, transp_mask_path))

        self.samples = samples
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        img_path, depth_path, valid_mask_path, transp_mask_path = self.samples[idx]

        img = np.asarray(Image.open(img_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16).astype('float')*50.0/65535

        mask_valid = np.asarray(Image.open(valid_mask_path))
        mask_transp = np.asarray(Image.open(transp_mask_path))

        depth = depth * mask_valid * mask_transp



        img = torch.from_numpy(img).permute(2,0,1)
        img = self.normalize(img)
        depth = torch.from_numpy(depth).unsqueeze(0)
        return dict(image=img, depth=depth, image_path=img_path, depth_path=depth_path)

    def __len__(self):
        return len(self.samples)


def get_ibims_loader(batch_size=1):
    dataloader = DataLoader(iBims(), batch_size=batch_size)
    return dataloader


@torch.no_grad()
def infer(model, images):
    # images.shape = N, C, H, W
    pred1 = model(images)[-1]
    pred2 = model(torch.flip(images, [3]))[-1]
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for sample in tqdm(test_loader):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        pred = infer(model, image)
        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, config=config))

    if round_vals:
        r = lambda m: round(m, round_precision)
    else:
        r = lambda m: m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics


def main(config):
    model = build_model(config)
    test_loader = get_ibims_loader()

    model = load_wts(model, config.checkpoint)
    # model.load_state_dict(torch.load(config.checkpoint))
    model = model.cuda()

    metrics = evaluate(model, test_loader, config)
    print(metrics)
    return metrics


def eval_by_id_pattern(model_name, pattern: str, checkpoint_dir="./checkpoints/", dataset='nyu', ckpt_type='best',
                       **kwargs):
    matches = glob.glob(os.path.join(checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
    if not (len(matches) > 0):
        raise ValueError(f"No matches found for the pattern {pattern}")

    checkpoint = matches[0]
    print(f"Evaluating {checkpoint} ...")
    config = get_config(model_name, dataset, checkpoint=checkpoint, **kwargs)
    pprint(config)
    return main(config)