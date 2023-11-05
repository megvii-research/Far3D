import numpy as np
import pandas as pd
from av2.geometry.geometry import mat_to_xyz, quat_to_mat, wrap_angles
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset, build_dataloader
import mmcv
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import importlib
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
result_path = 'test/dn_n3/Fri_Jul_21_13_46_06_2023.feather'
df = pd.read_feather(result_path) #['log_id', 'timestamp_ns', 'tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz', 'score', 'category'],

numpy_array = df.values
result = np.array(numpy_array)
save_pth = "./vis_attn/"
plugin_dir = 'projects/mmdet3d_plugin/'
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]

for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m
print(_module_path)
plg_lib = importlib.import_module(_module_path)
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
               'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
               'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
               'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
               'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
               'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER']
point_cloud_range = [-152.4, -152.4, -5.0, 152.4, 152.4, 5.0]
ida_aug_conf = {
        "resize_lim": (0.47, 0.55),
        "final_dim": (640, 960),
        "final_dim_f": (640, 720),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "rand_flip": False,
    }
vis_pipeline = [
    dict(type='AV2LoadMultiViewImageFromFiles', to_float32=True,
         use_nori=True, nori_pkl_name='s3://argoverse/nori/0410_camera/argoverse2_val_camera.pkl',
         ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='AV2ResizeCropFlipRotImageV2', data_aug_conf=ida_aug_conf, training=False),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d','lidar_timestamp'))
]

data_root = 'data/av2/'
pkl_path = 'av2_val_infos_mini.pkl'
vis_config = dict(
    type = 'Argoverse2DatasetT',
    data_root = data_root,
    collect_keys= ['img', 'img_metas'], 
    queue_length=1, 
    ann_file=data_root + pkl_path, 
    split='val',
    load_interval=1,
    classes=class_names, 
    interval_test=True,
    pipeline=vis_pipeline,
)
dataset = build_dataset(vis_config)
strides = [8, 16, 32, 64]
level = 0

if not os.path.exists(save_pth):
    os.mkdir(save_pth)
# for num in tqdm(range(1)):
for num in tqdm(range(len(dataset))):
    if num % 3 != 0:
        continue
    data = dataset[num]
    scene_token = data['img_metas'].data['scene_token']
    all_points_3d = torch.tensor(np.load('out_points_3d/{}.npy'.format(scene_token))) #(7, 965, 8, 4, 13, 2) #torch.Size([1, 942, 13, 3])
    weights = torch.tensor(np.load('out_weights/{}.npy'.format(scene_token))) #(7, 965, 8, 52)
    cls_scores = torch.tensor(np.load('cls_score/{}.npy'.format(scene_token))) #(1, 965, 26)
    cls_scores = cls_scores.sigmoid().max(-1)[0]
    scores, indexs = cls_scores.view(-1).topk(5)


    for stage in range(6):
        # if stage != 5:
        #     continue
        points_3d_stage = all_points_3d[stage].squeeze(0)#torch.Size([num, 13, 3])
        weight_stage = weights[stage]

        for idx in range(7):
            blk = np.zeros(img.shape, np.uint8) 
            img = data['img'].data[0][idx]
            lidar2img =data['lidar2img'].data[idx]
            img = np.ascontiguousarray(img.numpy().transpose(1, 2, 0))
            points_3d = torch.cat((points_3d_stage, torch.ones_like(points_3d_stage[..., :1])), dim=-1).reshape(-1, 4)
            points = torch.matmul(points_3d, lidar2img.T) #torch.Size([num, 13, 4])
            H, W, _ = img.shape
            # points = points_3d_stage[idx].mean(dim=1).permute(1, 0, 2, 3) #(4, num, 13, 2)
            points[..., :2] /= points[..., 2:3]
            points = points.reshape(-1, 13, 4)
            weight = weight_stage[idx].sum(dim=1).reshape(-1, 4, 13).permute(1, 0, 2) #(num, 52)->(4, num, 13)
            # points[..., 0:1] = points[..., 0:1].clip(0, W)
            # points[..., 1:2] = points[..., 1:2].clip(0, H)
            sub_xy_scale = points
            for level in range(4):
                attn_weight_scale = weight[level]
                sub_xys = sub_xy_scale[indexs]
                attn_weights = attn_weight_scale[indexs]
                for sub_xy, attn_weight in zip(sub_xys, attn_weights):
                    for j in range(len(sub_xy)):
                        if (sub_xy[j, 0] > 0 and sub_xy[j, 0] < W) and (sub_xy[j, 1] > 0 and sub_xy[j, 1] < H):
                            cv2.circle(blk, (int(sub_xy[j, 0]), int(sub_xy[j, 1])), 10, (255, int(500*attn_weight[j]), 0), -1)
                img = cv2.addWeighted(img, 0.7, blk, 0.8, 1, dtype = cv2.CV_32F)
                cv2.imwrite(f"{save_pth}data_{scene_token}_stage_{stage}_level_{level}_img_{idx}.jpg", img)
    