# NOT Used right now

import os
import glob
from natsort import natsorted
import re
import numpy as np
from tqdm import tqdm
from typing import List, AnyStr


# os.chdir(os.path.join(globals()['_dh'][0], "../"))
# files_path = "train_test_inputs/kitti_eigen_train_files_with_gt.txt"
# files_path_test = "train_test_inputs/kitti_eigen_test_files_with_gt.txt"
# data_root = "../dataset/kitti/raw"

# with open(files_path) as f:
#     filenames = f.readlines()

# with open(files_path_test) as f:
#     filenames_test = f.readlines()

def get_drive_name(s: str):
    """
    Extracts KITTI unique drive name from a given string containing path(s) to color or ground truth files
    Example 1: '2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png' --> 2011_09_26_drive_0001_sync
    Example 2: '2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png 2011_09_26_drive_0001_sync/projdepth/groundtruth/image_02/data/0000000005.png 731.2' --> 2011_09_26_drive_0001_sync
    """
    pattern = r"(\w+drive\w+)"
    return re.search(pattern, s).group(1)

def get_frame_number(line: str):
    """
    Get frame number from a line of filenames.txt
    """
    return int(os.path.basename(line.split()[0]).split(".")[0])

def all_same_drive(x: List[str]):
    return len(set(map(get_drive_name, x))) == 1

def all_consecutive(x: List[str]):
    return np.all(np.diff([get_frame_number(i) for i in x]) == 1)

def n_tuples(filenames: List[str], n: int):
    """
    Get n-tuples where all tuples satisfy:
        1. All elements of the tuple are from same KITTI drive
        2. all elements of the tuple are consecutive frames
    """
    sorted_filenames = natsorted(filenames, key = lambda x : (get_drive_name(x), get_frame_number(x)))
    for i in range(n-1, len(sorted_filenames)):
        elems = list(sorted_filenames[i-n-1: i-1])
        if all_same_drive(elems) and all_consecutive(elems):
            yield elems