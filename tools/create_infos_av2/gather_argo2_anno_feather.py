# "create complete infos of gts for evaluation"

import os
from os import path as osp
from av2.utils.io import read_feather
import pandas as pd   
from pathlib import Path     


if __name__ == '__main__':
    save_path = "/data/av2/val_anno.feather"
    split_dir = Path("/data/av2/val")
    annotations_path_list = split_dir.glob("*/annotations.feather")

    seg_anno_list = []
    for annotations_path in annotations_path_list:
        
        seg_anno = read_feather(Path(annotations_path))
        log_dir = os.path.dirname(annotations_path)
        log_id = log_dir.split('/')[-1]
        seg_anno["log_id"] = log_id
        seg_anno_list.append(seg_anno)
    
    gts = pd.concat(seg_anno_list).reset_index()
    gts.to_feather(save_path)