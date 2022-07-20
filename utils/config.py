import copy

from easydict import EasyDict as edict


AVAIL_BACKBONES = [

    # EfficientNet
    'tf_efficientnet_b5_ap',

    # EfficientNet-V2
    'tf_efficientnetv2_l',
    'tf_efficientnetv2_m',
    'tf_efficientnetv2_s',

    # MobileNet-V2
    'mobilenetv2_100',
    "densenet161",
 ]

COMMON_CONFIG = {
    "save_dir": "./checkpoints",
    "project": "LOCALBINS-NYU",
    "notes": "",
    "gpu": None,
    "root": "."
    }

DATASETS_CONFIG = {
    "kitti": {
        "dataset": "kitti",
        "min_depth": 0.001,
        "max_depth": 80,
        "data_path": "../dataset/kitti/raw",
        "gt_path": "../dataset/kitti/gts",
        "filenames_file": "./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
        "input_height": 352,
        "input_width": 704,
        "data_path_eval": "../dataset/kitti/raw",
        "gt_path_eval": "../dataset/kitti/gts",
        "filenames_file_eval": "./train_test_inputs/kitti_eigen_test_files_with_gt.txt",
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,

        "do_random_rotate": True,
        "degree": 1.0,
        "do_kb_crop": True,
        "garg_crop": True,
        "eigen_crop": False,
        "use_right": False
    },
    "kitti_test": {
        "dataset": "kitti",
        "min_depth": 0.001,
        "max_depth": 80,
        "data_path": "../dataset/kitti/raw",
        "gt_path": "../dataset/kitti/gts",
        "filenames_file": "./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
        "input_height": 352,
        "input_width": 1216,
        "data_path_eval": "../dataset/kitti/raw",
        "gt_path_eval": "../dataset/kitti/gts",
        "filenames_file_eval": "./train_test_inputs/kitti_eigen_test_files_with_gt.txt",
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,

        "do_random_rotate": False,
        "degree": 1.0,
        "do_kb_crop": False,
        "garg_crop": True,
        "eigen_crop": False,
        "use_right": False
    },
    "nyu": {
        "dataset": "nyu",
        "avoid_boundary": True,
        "min_depth": 1e-3,   # originally 0.1 
        "max_depth": 10, 
        "data_path": "../dataset/nyu_depth_v2/sync/",
        "gt_path": "../dataset/nyu_depth_v2/sync/",
        "filenames_file": "./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
        "input_height": 480,
        "input_width": 640,
        "data_path_eval": "../dataset/nyu_depth_v2/official_splits/test/",
        "gt_path_eval": "../dataset/nyu_depth_v2/official_splits/test/",
        "filenames_file_eval": "./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
        "min_depth_eval": 1e-3,
        "max_depth_eval": 10,

        "do_random_rotate": True,
        "degree": 1.0,
        "do_kb_crop": False,
        "garg_crop": False,
        "eigen_crop": True
    }
}


MODELS_CONFIGS = {
      "localbins":{
        "name": "LocalBins",
        "n_bins": 256,
        "version_name":"v4_1",
        "with_positional_encodings": False,
        "backbone" : "tf_efficientnet_b5_ap",
        "bin_embedding_dim": 128,
        "local_win_sizes": "3,7,15,31,63",
        "splitter_type": "linear"

        }, 

        # other models here ... 
  
}

TRAINING_CONFIG = {
    "localbins_h": {
        "epochs": 10,
        "bs": 16,
        "nf": 1,
        "optim_kwargs": {"lr": 0.000357, "wd": 0.1},
        "sched_kwargs": {"div_factor": 1, "final_div_factor": 10000, "pct_start": 0.7, "three_phase":False, "cycle_momentum": True},
        "same_lr": True,
        "w_chamfer": 0.02,  
        "w_si": 1,
        "avoid_boundary": False,
        "random_crop": False,
        "n_bbox_queries":200,
        "roi_align_mode":-1, 
        "input_width": 640,
        "input_height": 480,
        "init_temp": 1.,
        "final_temp": 1.,
        "loss_wts_scale":"0,0,0,1", 
        "gamma_window": 0.3, 
        "gamma_scale" : 0.3,
    }, 
    # other models here ...

}

COMMON_TRAINING_CONFIG = {
    "dataset": "nyu",
    "distributed": True,
    "workers": 16,
    "clip_grad" : 0.1,
    "use_shared_dict": False, 
    "shared_dict": None,
    "use_amp": False,

    "aug": True,
    "random_crop": True,

    "validate_every": 0.25,
    "log_images_every": 0.1,
    "prefetch": False, 
    "cut_depth": False
}



def flatten(config):
    def recurse(inp):
        if isinstance(inp, dict):
            for key, value in inp.items():
                if isinstance(value, dict):
                    yield from recurse(value)
                else:
                    yield (key, value)

    return dict(list(recurse(config)))
    

# older code
KEYS_TYPE_BOOL = ["use_amp", "distributed", "use_shared_dict", "same_lr", "aug", "three_phase", "prefetch", "use_gt_nudged_view", "flow_logw", "use_l2_flow", "cycle_momentum", "with_positional_encodings",
                    "cut_depth"]
KEYS_TYPE_INT = ["input_width", "input_height", "roi_align_mode"]  # Not necessary, handled by infer_type

def get_config(model_name, dataset, **overwrite_kwargs):
    # if 'use_config_from' in overwrite_kwargs:
    #     model_name = overwrite_kwargs['use_from_from']

    def get_training_config():
        tconf = TRAINING_CONFIG[model_name]
        if isinstance(tconf, str) and tconf.startswith("self"):
            key = tconf[5:]
            if not key in TRAINING_CONFIG:
                raise ValueError(f"Cannot resolve self reference {tconf}")
            print(f"Training configuration resolved to {key} config")
            tconf = TRAINING_CONFIG[key]
        return tconf

    config = {**COMMON_CONFIG, **COMMON_TRAINING_CONFIG, **DATASETS_CONFIG[dataset], **MODELS_CONFIGS[model_name],  **get_training_config()}
    config = flatten(config)
    config = {**config, **overwrite_kwargs}
    for key in KEYS_TYPE_BOOL:
        if key in config:
            config[key] = bool(config[key])

    for key in KEYS_TYPE_INT:
        if key in config:
            config[key] = int(config[key])

    if 'backbone' in config:
        assert config['backbone'] in AVAIL_BACKBONES, f"Unknown backbone {config['backbone']}. Available backbone models are : {AVAIL_BACKBONES}"

    if dataset == 'nyu':
        config['project'] = "LOCALBINS-NYU"
    if dataset == 'kitti':
        config['project'] = "LOCALBINS-KITTI"

    if 'gamma_scale' in config:
        loss_wts = [str(config['gamma_scale']**i) for i in range(4)][::-1]
        config['loss_wts_scale'] = ",".join(loss_wts)
        
    return edict(config)
