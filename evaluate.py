from data.data_mono import DepthDataLoader
from utils.misc import compute_metrics, RunningAverageDict, count_parameters
from utils.config import get_config
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
        r = lambda m : round(m, round_precision)
    else:
        r = lambda m : m
    metrics = {k: r(v) for k,v in metrics.get_value().items()}
    return metrics



def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, 'online_eval').data

    model = load_wts(model, config.checkpoint)
    # model.load_state_dict(torch.load(config.checkpoint))
    model = model.cuda()

    metrics = evaluate(model, test_loader, config)
    print(metrics)
    metrics['#params'] = f"{round(count_parameters(model)/1e6, 2)}M"
    return metrics


def eval_by_id_pattern(model_name, pattern: str, checkpoint_dir="./checkpoints/", dataset='nyu', ckpt_type='best', **kwargs):
    matches = glob.glob(os.path.join(checkpoint_dir,f"*{pattern}*{ckpt_type}*"))
    if not (len(matches) > 0):
        raise ValueError(f"No matches found for the pattern {pattern}")

    checkpoint = matches[0]
    print(f"Evaluating {checkpoint} ...")
    config = get_config(model_name, dataset, checkpoint=checkpoint, **kwargs)
    pprint(config)
    return main(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="adnet")
    parser.add_argument("--dataset", type=str, default='nyu')
    parser.add_argument("--checkpoint", type=str, required=True)

    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)
    config = get_config(args.model, args.dataset,model=args.model, checkpoint=args.checkpoint, **overwrite_kwargs)

    # config = get_config(args.model, args.dataset, checkpoint=args.checkpoint, model=args.model)

    main(config)



