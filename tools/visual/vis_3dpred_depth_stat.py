import torch
from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset
import os
import importlib
import torchvision
import numpy as np
import cv2
from PIL import ImageDraw
from PIL import Image
import sys
from mmdet3d.models import build_detector
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.core import MlvlPointGenerator
# from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
import matplotlib

'''
进行以下所需的改动：
1. petr3d.py forward_test  if key != 'img' and key != 'gt_bboxes_3d' and key != 'gt_labels_3d'  
2.argoverse2_dataset_t.py  注释if not self.test_mode  
3. config  test_pipeline加上LoadAnnotations3D， keys加上'gt_bboxes_3d', 'gt_labels_3d'  
4.可以使用mini pkl
5. If needed, set vis_return_ref3d=True in config; and modify below receiving part.
6. 如果使用 depths, gt_2dbox 等，需要再 AV2Resize里注释掉 self.training，并在 keys 中添加上 'ins_depthmap' 等
7. 使用 ref_dict 的话在本文件中加入 ref_dict 相关，及 petr3d.py 中恢复 ref_dict 相关
'''

colors = matplotlib.cm.get_cmap("plasma")
col = colors(np.linspace(0, 1, 26))

flag_use_server = True
if flag_use_server:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

score =  0.30    # 0.3-0.35 TODO
config = 'projects/configs/0516/stream_petrv2_yolox_insdepth4c_th1e-1_vispred.py'
    # 'projects/configs/a100/stream_petrv2_seq_v2_eva.py'
    # 'projects/configs/0506/stream_petrv2_yolo2dp_th1e-1_vispred_gtdepth.py'
    # '/data/PETR-stream/projects/configs/yolox/yolox_pre.py'
checkpoint = 'work_dirs/stream_petrv2_yolox_insdepth4c_un1/latest.pth'
    # 'work_dirs/stream_petrv2_seq_v2_eva/latest.pth'
    # 'work_dirs/stream_petrv2_yolo2dp_th1e-1/latest.pth'
    # '/data/PETR-stream/work_dirs/yolox/yolox_pre/latest.pth'
flag_vis_ref_point = True
print('Use config: ', config)

cfg = Config.fromfile(config)
cfg.model.pretrained = None
cfg.data.test.test_mode = True
plugin_dir = cfg.plugin_dir
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]
for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m
sys.path.append(os.getcwd())
plg_lib = importlib.import_module(_module_path)

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)
# cfg.model.train_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location='cpu')
# memory = []
# hook = [model.pts_bbox_head.register_forward_hook(
#         lambda self, input, output: memory.append(output))]
pc_range = torch.tensor([-152.4, -152.4, -5.0, 152.4, 152.4, 5.0]).cuda()
model = MMDataParallel(model, device_ids=[0])
model.eval()

ref_dict = dict()
depth_pred = []
depth_gt = []
for k, data in enumerate(data_loader):
    with torch.no_grad():
        out, ref_dict = model(return_loss=False, rescale=True, **data)      # TODO vis rollback (, ref_dict)
        depth_pred.append(ref_dict['pred_depth'].flatten())
        model.module.pts_bbox_head.val_use_gt_depth = True
        out, ref_dict = model(return_loss=False, rescale=True, **data)  # TODO vis rollback (, ref_dict)
        depth_gt.append(ref_dict['pred_depth'].flatten())
        model.module.pts_bbox_head.val_use_gt_depth = False

depth_pred = torch.cat(depth_pred, dim=0)
depth_gt = torch.cat(depth_gt, dim=0)

# mean and var
abs_diff = torch.abs(depth_pred - depth_gt)
mean_diff = torch.mean(abs_diff).cpu().numpy()
var_diff = torch.var(abs_diff).cpu().numpy()
print("Mean of absolute difference:", mean_diff)
print("Variance of absolute difference:", var_diff)

# plot
depth_gt = depth_gt.cpu().numpy()
depth_gt_int = depth_gt.astype(np.int32)
depth_pred = depth_pred.cpu().numpy()
diff = depth_pred - depth_gt
plt.figure(figsize=(10,10))
plt.scatter(depth_gt_int, depth_gt, c='r', marker="s", label='gt')
plt.scatter(depth_gt_int, depth_pred, c='b', marker="o", label='pred')
for ith in range(len(diff)):
    plt.text(depth_gt_int[ith], depth_pred[ith], '%d'%diff[ith])
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.title("Depth Pred Difference: mean %.1f, std %.1f" % (mean_diff, var_diff))
plt.savefig('vis/vis_depth_gap.png')


def _convert_bin_depth_to_specific(pred_indices):
    depth_min, depth_max, num_bins = 1e-3, 110, 50
    bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
    # indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    depth = depth_min + bin_size / 8 * (torch.square(pred_indices / 0.5 + 1) - 1)
    return depth