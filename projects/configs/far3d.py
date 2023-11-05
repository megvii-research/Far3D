_base_ = [
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-152.4, -152.4, -5.0, 152.4, 152.4, 5.0]

voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
               'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
               'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
               'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
               'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
               'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER']

num_gpus = 8
batch_size = 1
num_iters_per_epoch = 110071 // (num_gpus * batch_size) 
num_epochs = 6
embed_dims=256

queue_length = 1
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
depthnet_config = {'type': 0, 'hidden_dim': 256, 'num_depth_bins': 50, 'depth_min': 1e-1, 'depth_max': 110, 'stride': 8}
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Far3D',
    use_grid_mask=True,
    stride=[8, 16, 32, 64],
    position_level=[0, 1, 2, 3],
    img_backbone=dict(
        type='VoVNet', ###use checkpoint to save memory
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage2','stage3','stage4','stage5',)),
    img_neck=dict(
        type='FPN',  ###remove unused parameters 
        start_level=1,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        num_outs=4),
    img_roi_head=dict(
        type='YOLOXHeadCustom',
        num_classes=26,
        in_channels=256,
        strides=[8, 16, 32, 64],
        train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
        pred_with_depth=True,
        depthnet_config=depthnet_config,
        reg_depth_level='p3',
        pred_depth_var=False,    # note 2d depth uncertainty
        loss_depth2d=dict(type='L1Loss', loss_weight=1.0),
        sample_with_score=True,  # note threshold
        threshold_score=0.1,
        topk_proposal=None,
        return_context_feat=True,
    ),
    pts_bbox_head=dict(
        type='FarHead',
        num_classes=26,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        offset=0.5,
        offset_p=0.0,
        num_smp_per_gt=3,
        with_dn=True,
        with_ego_pos=True,
        add_query_from_2d=True,
        pred_box_var=False,  # note add box uncertainty
        depthnet_config=depthnet_config,
        train_use_gt_depth=True,
        add_multi_depth_proposal=True,
        multi_depth_config={'topk': 1, 'range_min': 30,},  # 'bin_unit': 1, 'step_num': 4,
        return_bbox2d_scores=True,
        return_context_feat=True,
        code_size=8,
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                embed_dims=256,
                num_layers=6,
                transformerlayers=dict(
                    type='Detr3DTemporalDecoderLayer',
                    batch_first=True,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='DeformableFeatureAggregationCuda', 
                            embed_dims=256,
                            num_groups=8,
                            num_levels=4,
                            num_cams=7,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=26), 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range),)))


dataset_type = 'Argoverse2DatasetT'
data_root = 'data/av2/'

file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.47, 0.55),
        "final_dim": (640, 960),
        "final_dim_f": (640, 720),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "rand_flip": False,
    }
train_pipeline = [
    dict(type='AV2LoadMultiViewImageFromFiles', to_float32=True,
         use_nori=True, nori_pkl_name='s3://argoverse/nori/0410_camera/argoverse2_train_camera.pkl',
         ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='AV2ResizeCropFlipRotImageV2', data_aug_conf=ida_aug_conf),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='AV2PadMultiViewImage', size='same2max'),
    dict(type='AV2DownsampleQuantizeInstanceDepthmap', downsample=depthnet_config['stride'], depth_config=depthnet_config),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d', 'ins_depthmap', 'ins_depthmap_mask'))
]
test_pipeline = [
    dict(type='AV2LoadMultiViewImageFromFiles', to_float32=True,
         use_nori=True, nori_pkl_name='s3://argoverse/nori/0410_camera/argoverse2_val_camera.pkl',
         ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='AV2ResizeCropFlipRotImageV2', data_aug_conf=ida_aug_conf),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='AV2PadMultiViewImage', size='same2max'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_bboxes', 'centers2d'] + collect_keys,
            meta_keys=('filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'av2_train_infos.pkl',
        split='train',
        load_interval=1,
        num_frame_losses=num_frame_losses,
        seq_split_num=2,
        seq_mode=True,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=False,
        interval_test=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        data_root=data_root,
        collect_keys=collect_keys + ['img', 'img_metas'], 
        queue_length=queue_length, 
        ann_file=data_root +  'av2_val_infos.pkl', 
        split='val',
        load_interval=1,
        classes=class_names, 
        modality=input_modality,
        interval_test=True,),
    test=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        data_root=data_root,
        collect_keys=collect_keys + ['img', 'img_metas'], 
        queue_length=queue_length, 
        ann_file=data_root +  'av2_val_infos.pkl', 
        split='val',
        load_interval=1,
        classes=class_names, 
        modality=input_modality,
        interval_test=True,),

        shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
        nonshuffler_sampler=dict(type='DistributedSampler'),
    )

optimizer = dict(
    type='AdamW', 
    lr=2e-4, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1), # 0.5 only for Focal-PETR with R50-in1k pretrained weights
        }),
    weight_decay=0.01)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)
custom_hooks = [dict(type="UseGtDepthHook", stop_gt_depth_iter=22000)]
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from='ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from=None

# fps: 6.4 img / s
