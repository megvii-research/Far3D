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

score =  0.2    # 0.3-0.35 TODO
config = 'projects/configs/0516/stream_petrv2_yolox_insdepth4c_th1e-1_vispred.py'
    # 'projects/configs/0530/stream_petrv2_yolox_insdepth4c_md6_vispred.py'
    # 'projects/configs/a100/stream_petrv2_seq_v2_eva.py'
    # 'projects/configs/0506/stream_petrv2_yolo2dp_th1e-1_vispred_gtdepth.py'
    # '/data/PETR-stream/projects/configs/yolox/yolox_pre.py'
checkpoint = 'work_dirs/stream_petrv2_yolox_insdepth4c_th1e-1/latest.pth'
    # 'work_dirs/stream_petrv2_yolox_insdepth4c_md6/latest.pth'
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
for k, data in enumerate(data_loader):
    with torch.no_grad():
        out, ref_dict = model(return_loss=False, rescale=True, **data)      # TODO vis rollback (, ref_dict)

    scores = out[0]['pts_bbox']['scores_3d']  # [300]
    cat_id = out[0]['pts_bbox']['labels_3d']  # [300]
    boxes_3d = out[0]['pts_bbox']['boxes_3d']
    corners_3d = boxes_3d.corners.reshape(-1, 8, 3)
    boxes_3d = torch.cat((boxes_3d.gravity_center, boxes_3d.tensor[:, 3:]), dim=1)  # [300, 7]

    gt_bboxes_3d = data['gt_bboxes_3d'][0].data[0][0]
    gt_labels_3d = data['gt_labels_3d'][0].data[0][0]  # [num, ]
    gt_corners_3d = gt_bboxes_3d.corners.reshape(-1, 8, 3)
    gt_bboxes_3d = torch.cat((gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1)  # [num, 7]

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
            y.append(bottom_corners_bev[i, j, 1])
        x.append(bottom_corners_bev[i, 0, 0])
        y.append(bottom_corners_bev[i, 0, 1])
        #     plt.plot(x, y, 'r^-', color=col[mask_cat_id[i]], linewidth=3)
        plt.plot(x, y, color='b', linewidth=3, label='pred_corner')
        plt.text(x[-1], y[-1], mask_cat_id[i].numpy(), fontsize=10)
        plt.text(x[-1]+1, y[-1]+1, format(mask_score[i].numpy(), ".2f"), fontsize=15)

    for i in range(len(gt_bottom_corners_bev)):
        x = []
        y = []
        for j in range(len(gt_bottom_corners_bev[i])):
            x.append(gt_bottom_corners_bev[i, j, 0])
            y.append(gt_bottom_corners_bev[i, j, 1])
        x.append(gt_bottom_corners_bev[i, 0, 0])
        y.append(gt_bottom_corners_bev[i, 0, 1])
        plt.plot(x, y, color='r', linewidth=3, label='gt_corner')
        plt.text(x[-1], y[-1], gt_labels_3d[i].numpy(), fontsize=10)
        #     plt.plot(x, y, 'r^-', color=col[gt_labels_3d[i]], linewidth=3)

    if flag_vis_ref_point:
        ref_3d, ref_2d = ref_dict['ref3d'], ref_dict['ref2d']
        ref_3d_scale = ref_3d[..., :2] * (pc_range[3:5] - pc_range[0:2]) + pc_range[0:2]
        ref_2d_scale = ref_2d[..., :2] * (pc_range[3:5] - pc_range[0:2]) + pc_range[0:2]
        x2npy = lambda x: x.cpu().detach().numpy()
        ref_3d_scale, ref_2d_scale = x2npy(ref_3d_scale), x2npy(ref_2d_scale)
        plt.scatter(ref_3d_scale[..., 0], ref_3d_scale[..., 1], c='c', marker='o')
        plt.scatter(ref_2d_scale[..., 0], ref_2d_scale[..., 1], c='g', s=200, marker='D', label='ref2d')
        import ipdb; ipdb.set_trace()
        valid_indices = np.logical_and(np.abs(ref_3d_scale[..., 0])<=100, np.abs(ref_3d_scale[..., 1])<=100)

    # Increase the font size of the axis labels and tick labels
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.title('2D proposal (Green) num: %d ; GT (Red) num: %d; Pred (Blue) num: %d ' % (len(ref_2d), len(gt_bottom_corners_bev), len(mask_bottom_corners_bev)), fontsize = 40)
    plt.savefig('vis/vis_3dpred_insdepth4c_md6/{}-{}-2d.png'.format(k, score))   # TODO
    # plt.show()
    plt.clf()
