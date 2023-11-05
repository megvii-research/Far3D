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
#红色为pred，绿色为gt
idx = 13
score_thr = 0.25
num_samples = 1500
pkl_path = 'av2_val_infos_mini.pkl'
save_pth = "./vis/"
result_path = 'test/dn_n3/Sat_Aug_12_12_13_45_2023.feather'
df = pd.read_feather(result_path) #['log_id', 'timestamp_ns', 'tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz', 'score', 'category'],

numpy_array = df.values
result = np.array(numpy_array)


if not os.path.exists(save_pth):
    os.mkdir(save_pth)
plugin_dir = 'projects/mmdet3d_plugin/'
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]

for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m
print(_module_path)
plg_lib = importlib.import_module(_module_path)

collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
               'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
               'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
               'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
               'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
               'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER']
point_cloud_range = [-152.4, -152.4, -5.0, 152.4, 152.4, 5.0]

ida_aug_conf = {
        "resize_lim": (0.94, 1.1),
        "final_dim": (1280, 1920),
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
data_config={
    'input_size': (640, 960),

    # Augmentation
    'resize': (0.47, 0.55),
    'rot': (0.0, 0.0),
    'flip': True,
    'crop_h': (0.0, 0.0),
}
img_norm_cfg = dict(
    mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229] )

data_root = 'data/av2/'

vis_config = dict(
    type = 'Argoverse2DatasetT',
    data_root = data_root,
    collect_keys=collect_keys + ['img', 'img_metas'], 
    queue_length=1, 
    ann_file=data_root + pkl_path, 
    split='val',
    load_interval=1,
    classes=class_names, 
    interval_test=True,
    pipeline=vis_pipeline,
)

lines = [(0, 1), (1, 2), (2, 3), (3, 0), 
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)]

dataset = build_dataset(vis_config)

for data_idx in range(13, 14):
    # if data_idx % 50 != 0:
    #     continue
    data = dataset[data_idx]
    
    mask = result[:, 1] == data['img_metas'].data['lidar_timestamp']
    pred = result[mask]
    pred_score = result[mask][..., 12]
    pred_class = result[mask][..., 13]

    score_mask = pred_score > score_thr
    pred_class = pred_class[score_mask]
    pred = pred[score_mask]
    yaw = mat_to_xyz(quat_to_mat(pred[..., 8:12]))[..., 2:3]

    pred_bbox =np.concatenate([pred[..., 2:8], yaw], axis=1).astype("float32")
    pred_bbox = torch.from_numpy(pred_bbox)

    pred_bboxes = LiDARInstance3DBoxes(
        pred_bbox, #xyzlwh+yaw
        box_dim=pred_bbox.shape[-1],
        origin=(0.5, 0.5, 0.5))

    for idx in range(7):

        img = data['img'][idx]
        print(img)
        # img = data['img'].data[0][idx]

        lidar2img =data['lidar2img'].data[0][idx]

        pred_bboxes_3d = pred_bboxes.corners
        pred_bboxes_points = pred_bboxes_3d.view(-1, 3)
        pred_bboxes_points = np.concatenate((pred_bboxes_points[:, :3], np.ones(pred_bboxes_points.shape[0])[:, None]), axis=1)

        pred_bboxes_points = np.matmul(pred_bboxes_points, lidar2img.T) #(cam_intrinsic @ lidar2cam).T)
        pred_bboxes_points[:, :2] /= pred_bboxes_points[:, 2:3]
        img = np.ascontiguousarray(img.numpy().transpose(1, 2, 0))
        bboxes = pred_bboxes_points.reshape(-1, 8, 4)

        for index, bbox in enumerate(bboxes, 0):
            for line in lines:
                if(bbox[line[0]][2] > 0 and bbox[line[1]][2] > 0):
                    if pred_score[index]< 0.15:
                        img = cv2.line(img, 
                                    (int(bbox[line[0]][0]), int(bbox[line[0]][1])),
                                    (int(bbox[line[1]][0]), int(bbox[line[1]][1])), (100, 255, 0), 2)
                    else:
                        img = cv2.line(img, 
                                    (int(bbox[line[0]][0]), int(bbox[line[0]][1])),
                                    (int(bbox[line[1]][0]), int(bbox[line[1]][1])), (255, 100, 0), 2)

        gt_bboxes_3d = data['gt_bboxes_3d'].data.corners
        gt_bboxes_points = gt_bboxes_3d.view(-1, 3)
        gt_bboxes_points = np.concatenate((gt_bboxes_points[:, :3], np.ones(gt_bboxes_points.shape[0])[:, None]), axis=1)

        gt_bboxes_points = np.matmul(gt_bboxes_points, lidar2img.T) #(cam_intrinsic @ lidar2cam).T)
        gt_bboxes_points[:, :2] /= gt_bboxes_points[:, 2:3]

        bboxes = gt_bboxes_points.reshape(-1, 8, 4)

        for index, bbox in enumerate(bboxes, 0):
            for line in lines:
                if(bbox[line[0]][2] > 0 and bbox[line[1]][2] > 0):
                    img = cv2.line(img, 
                                (int(bbox[line[0]][0]), int(bbox[line[0]][1])),
                                (int(bbox[line[1]][0]), int(bbox[line[1]][1])), (0, 100, 255), 3)



        cv2.imwrite(f"{save_pth}data_{data_idx}_img_{idx}.jpg", img)

    show_range = 150  # Range of visualization in BEV
    canva_size = 2000  # Size of canva in pixel
    canvas = np.ones((int(canva_size), int(canva_size), 3),
                        dtype=np.uint8)
    canvas *= 255

    corners_lidar = data['gt_bboxes_3d'].data.corners.reshape(-1, 8, 3).numpy()
    corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
    bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
    bottom_corners_bev = \
        (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
    center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
    head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
    canter_canvas = \
        (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas = canter_canvas.astype(np.int32)
    head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas = head_canvas.astype(np.int32)
    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for rid in range(len(head_bev)):
        for index in draw_boxes_indexes_bev:
            cv2.line(
                canvas,
                bottom_corners_bev[rid, index[0]],
                bottom_corners_bev[rid, index[1]],
                (0, 100, 255),
                thickness=2)
        cv2.line(
            canvas,
            center_canvas[rid],
            head_canvas[rid],
            (0, 100, 255),
            2,
            lineType=8)

            # draw instances
    corners_lidar = pred_bboxes.corners.reshape(-1, 8, 3).numpy()
    corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
    bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
    bottom_corners_bev = \
        (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
    center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
    head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
    canter_canvas = \
        (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas = canter_canvas.astype(np.int32)
    head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas = head_canvas.astype(np.int32)
    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for rid in range(len(head_bev)):
        if pred_score[rid]< 0.15:
            for index in draw_boxes_indexes_bev:
                cv2.line(
                    canvas,
                    bottom_corners_bev[rid, index[0]],
                    bottom_corners_bev[rid, index[1]],
                    (100, 255, 0),
                    thickness=3)
            cv2.line(
                canvas,
                center_canvas[rid],
                head_canvas[rid],
                (100, 255, 0),
                2,
                lineType=8)
        else:
            for index in draw_boxes_indexes_bev:
                cv2.line(
                    canvas,
                    bottom_corners_bev[rid, index[0]],
                    bottom_corners_bev[rid, index[1]],
                    (255, 100, 0),
                    thickness=3)
            cv2.line(
                canvas,
                center_canvas[rid],
                head_canvas[rid],
                (255, 100, 0),
                2,
                lineType=8)

    cv2.imwrite(f"{save_pth}data_{data_idx}_bev_img.jpg", canvas)