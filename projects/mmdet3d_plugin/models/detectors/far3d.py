# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
@DETECTORS.register_module()
class Far3D(MVXTwoStageDetector):
    """Far3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 depth_branch=None,
                 stride=[16],
                 position_level=[0],
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None):
        super(Far3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None


    def extract_img_feat(self, img, return_depth=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for i in self.position_level:
            BN, C, H, W = img_feats[i].size()
            img_feat_reshaped = img_feats[i].view(B, int(BN/B), C, H, W)
            img_feats_reshaped.append(img_feat_reshaped)
        
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(img_feats_reshaped)
        else:
            depths = None

        if return_depth:
            return img_feats_reshaped, depths
        return img_feats_reshaped

    @auto_fp16(apply_to=('img', 'img_feats'), out_fp32=True)
    def extract_feat(self, img, return_depth=False):
        """Extract features from images and points."""
        img_feats, depths = self.extract_img_feat(img, return_depth)
        if return_depth:
            return img_feats, depths
        return img_feats



    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        assert len(self.stride) == len(data['img_feats'])
        location_r = []
        for i in range(len(data['img_feats'])):
            bs, n = data['img_feats'][i].shape[:2]
            x = data['img_feats'][i].flatten(0, 1)
            location = locations(x, self.stride[i], pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
            location_r.append(location)
        return location_r

    def forward_roi_head(self, location, **data):
        outs_roi = self.img_roi_head(location, **data)
        
        return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_metas, **data)

        outs_roi = self.forward_roi_head(location, **data)
        bbox_dict = self.img_roi_head.get_bboxes(outs_roi, **data)  # {'bbox_list': BN x (Mi, 4), 'bbox_score_list': BN x (Mi, 1)}
        outs_roi.update(bbox_dict)
        outs = self.pts_bbox_head(img_metas, outs_roi, **data)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        if self.with_img_roi_head:
            loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
            losses2d = self.img_roi_head.loss(*loss2d_inputs)
            losses.update(losses2d) 

            return losses

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        for key in data:
            if key != 'gt_depth':
                data[key] = data[key][:, 0] 
        rec_img = data['img']
        rec_img_feats, depths = self.extract_feat(rec_img, True)

        data['img_feats'] = rec_img_feats
        losses = self.forward_pts_train(gt_bboxes_3d,
                                    gt_labels_3d, gt_bboxes,
                                    gt_labels, img_metas, centers2d, depths, **data)
        if depths is not None and "gt_depth" in data:
            losses["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )

        return losses
  
    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key not in ['img', 'gt_bboxes_3d', 'gt_bboxes', 'centers2d']:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        bbox_dict = self.img_roi_head.get_bboxes(outs_roi)  # {'bbox_list': BN x (Mi, 4), 'bbox_score_list': BN x (Mi, 1)}
        bbox_roi = bbox_dict['bbox_list']
        outs_roi.update(bbox_dict)

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs = self.pts_bbox_head(img_metas, outs_roi, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results, bbox_roi
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'])

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts, bbox_roi = self.simple_test_pts(img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            # result_dict['bbox_roi'] = bbox_roi
        return bbox_list

    