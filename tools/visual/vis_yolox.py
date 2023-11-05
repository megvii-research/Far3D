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
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
import matplotlib

colors = matplotlib.cm.get_cmap("plasma")
col = colors(np.linspace(0, 1, 26))

score = 0.2

flag_use_server = True
if flag_use_server:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

config = '/data/PETR-stream/projects/configs/yolox/yolox_pre.py'
checkpoint = '/data/PETR-stream/work_dirs/yolox/yolox_pre/latest.pth'

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

model = MMDataParallel(model, device_ids=[0])
model.eval()

for k, data in enumerate(data_loader):
        with torch.no_grad():
              out = model(return_loss=False, rescale=True, **data)
        scores = out[0]['pts_bbox']['scores_3d'] #[300]
        cat_id = out[0]['pts_bbox']['labels_3d'] #[300]
        boxes_3d = out[0]['pts_bbox']['boxes_3d']
        corners_3d = boxes_3d.corners.reshape(-1, 8, 3)
        boxes_3d = torch.cat((boxes_3d.gravity_center, boxes_3d.tensor[:, 3:]),dim=1) #[300, 7]

        gt_bboxes_3d = data['gt_bboxes_3d'][0].data[0][0]
        gt_labels_3d = data['gt_labels_3d'][0].data[0][0] #[num, ]
        gt_corners_3d = gt_bboxes_3d.corners.reshape(-1, 8, 3)
        gt_bboxes_3d = torch.cat((gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),dim=1) #[num, 7]

        bottom_corners_bev = corners_3d[:, [0, 3, 7, 4], :2]
        gt_bottom_corners_bev = gt_corners_3d[:, [0, 3, 7, 4], :2]

        fig = plt.figure(figsize=(48, 24))
        mask = scores >= score
        mask_bottom_corners_bev = bottom_corners_bev[mask]
        mask_score = scores[mask]
        mask_cat_id = cat_id[mask]
        for i in range(len(mask_bottom_corners_bev)):
                x = []
                y = []
                for j in range(len(mask_bottom_corners_bev[i])):
                        x.append(bottom_corners_bev[i, j, 0])
                        y.append(bottom_corners_bev[i, j , 1])
                x.append(bottom_corners_bev[i, 0, 0])
                y.append(bottom_corners_bev[i, 0, 1])
                #     plt.plot(x, y, 'r^-', color=col[mask_cat_id[i]], linewidth=3)
                plt.plot(x, y, color='b', linewidth=3)
                plt.text(x[-1], y[-1], mask_cat_id[i].numpy(), fontsize=10)

        for i in range(len(gt_bottom_corners_bev)):
                x = []
                y = []
                for j in range(len(gt_bottom_corners_bev[i])):
                        x.append(gt_bottom_corners_bev[i, j, 0])
                        y.append(gt_bottom_corners_bev[i, j , 1])
                x.append(gt_bottom_corners_bev[i, 0, 0])
                y.append(gt_bottom_corners_bev[i, 0, 1])
                plt.plot(x, y, color='r', linewidth=3)
                plt.text(x[-1], y[-1], gt_labels_3d[i].numpy(), fontsize=10)
                #     plt.plot(x, y, 'r^-', color=col[gt_labels_3d[i]], linewidth=3)
        plt.savefig('/data/PETR-stream/vis/{}-{}-2d.png'.format(k, score))
        plt.show()

