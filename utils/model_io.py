import torch
import glob
import os


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    wts = ckpt['model']

    state = {}
    for k, v in wts.items():
        if k.startswith('module.'):
            k = k.replace('module.','')
        state[k] = v

    model.load_state_dict(state)
    return model


