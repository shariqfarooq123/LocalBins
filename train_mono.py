import argparse
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data.distributed

from data.data_mono import DepthDataLoader
from data.types import DataLoaderTypes
from utils.config import get_config
from utils.misc import parallelize, count_parameters
from utils.arg_utils import parse_unknown
from pprint import pprint

from trainers.builder import get_trainer, TrainerTypes
from models.builder import build_model


def infer_trainer_type(trainer_type):
    if trainer_type == "si":
        return TrainerTypes.SILOG
    
    if trainer_type == "silog_chamfer":
        return TrainerTypes.SILOG_CHAMFER

def fix_random_seed(seed: int):
    import random
    import torch
    import numpy

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main_worker(gpu, ngpus_per_node, config):
    try:
        fix_random_seed(42)

        config.gpu = gpu

        model = build_model(config)
        model = parallelize(config, model)

        print(f"Total parameters : {round(count_parameters(model)/1e6,2)}M")

        train_loader = DepthDataLoader(config, "train").data  
        test_loader = DepthDataLoader(config, "online_eval").data
        config.dataloader_type = DataLoaderTypes.DICT_MONO

        trainer = get_trainer(config, infer_trainer_type(config.trainer))(config, model, train_loader, test_loader, device=config.gpu)

        trainer.train()
    finally:
        import wandb;wandb.finish()



if __name__ == '__main__':
    mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="synunet")
    parser.add_argument("--dataset", type=str, default='nyu')
    parser.add_argument("--trainer", type=str, default=None)

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)  # TODO check if overwrite_kwargs arguments are defined in config or really "novel" / uknown
    config = get_config(args.model, args.dataset, **{**dict(model=args.model, trainer=args.trainer), **overwrite_kwargs})
    # git_commit()
    if config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    config.shared_dict = shared_dict

    config.batch_size = config.bs
    config.mode = 'train'
    config.chamfer = config.w_chamfer > 0
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])
        #config.save_dir = "/ibex/scratch/bhatsf/videodepth/checkpoints"

    except KeyError as e:
        # We are NOT using SLURM
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:

        print(config.rank)
        port = np.random.randint(15000, 15025)
        config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(config.dist_url)
        config.dist_backend = 'nccl'
        config.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)
