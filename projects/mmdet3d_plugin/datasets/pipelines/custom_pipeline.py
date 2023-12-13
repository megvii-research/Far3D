import mmcv
import copy
import torch
import numpy as np
from PIL import Image
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type
import torch.nn.functional as F
import pickle
import refile
import numpy as np
from io import BytesIO
import pyarrow
import pyarrow.feather as feather
import pandas

@PIPELINES.register_module()
class AV2LoadMultiViewImageFromFiles(object):
    
    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        
    def __call__(self, results):
        filename = results['img_filename']
        img = [mmcv.imread(name, self.color_type).astype(np.float32) for name in filename]

        results['filename'] = [str(name) for name in filename]
        results['img'] = img
        results['img_shape'] = img[0].shape
        results['ori_lidar2img'] = copy.deepcopy(results['lidar2img'])
        results['scale_factor'] = 1.0
        num_channels = 3
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
        
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class AV2ResizeCropFlipRotImageV2():
    def __init__(self, data_aug_conf=None, multi_stamps=False):
        self.data_aug_conf = data_aug_conf
        self.min_size = 2.0
        self.multi_stamps = multi_stamps

    def __call__(self, results):
        imgs = results['img']
        N = len(imgs) // 2 if self.multi_stamps else len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        ida_mats = []

        with_depthmap = ('depthmap' in results.keys())
        if with_depthmap:
            new_depthmaps = []

        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        for i in range(N):
            H, W = imgs[i].shape[:2]
            if H > W:
                resize, resize_dims, crop = self._sample_augmentation_f(imgs[i])
                img = Image.fromarray(np.uint8(imgs[i]))
                if with_depthmap:
                    depthmap_im = Image.fromarray(results['depthmap'][i])
                else:
                    depthmap_im = None
                img, ida_mat_f, depthmap = self._img_transform(img, resize=resize, resize_dims=resize_dims, crop=crop, depthmap=depthmap_im)
                img_f = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes'])>0:
                    gt_bboxes = results['gt_bboxes'][i]
                    centers2d = results['centers2d'][i]
                    gt_labels = results['gt_labels'][i]
                    depths = results['depths'][i]
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                            img_f, gt_bboxes, centers2d, gt_labels, depths, resize=resize, crop=crop,
                        )
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation(img_f)
                img = Image.fromarray(np.uint8(img_f))
                img, ida_mat, depthmap = self._img_transform(img, resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate, depthmap=depthmap)
                img = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes'])>0:
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(img, gt_bboxes, centers2d, gt_labels,depths, resize=resize, crop=crop, flip=flip,)
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(img, gt_bboxes, centers2d, gt_labels, depths)
                

                    new_gt_bboxes.append(gt_bboxes)
                    new_centers2d.append(centers2d)
                    new_gt_labels.append(gt_labels)
                    new_depths.append(depths)

                new_imgs.append(img)
                results['intrinsics'][i][:3, :3] = ida_mat @ ida_mat_f @ results['intrinsics'][i][:3, :3]
                ida_mats.append(np.array(ida_mat @ ida_mat_f))
                if with_depthmap:
                    new_depthmaps.append(np.array(depthmap))

            else:
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation(imgs[i])
                img = Image.fromarray(np.uint8(imgs[i]))
                if with_depthmap:
                    depthmap_im = Image.fromarray(results['depthmap'][i])
                else:
                    depthmap_im = None
                img, ida_mat, depthmap = self._img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                    depthmap=depthmap_im,
                )
                img = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes'])>0:
                    gt_bboxes = results['gt_bboxes'][i]
                    centers2d = results['centers2d'][i]
                    gt_labels = results['gt_labels'][i]
                    depths = results['depths'][i]
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                            img,
                            gt_bboxes,
                            centers2d,
                            gt_labels,
                            depths,
                            resize=resize,
                            crop=crop,
                            flip=flip,
                        )
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(
                            img,
                            gt_bboxes,
                            centers2d,
                            gt_labels,
                            depths
                        )
                    

                    new_gt_bboxes.append(gt_bboxes)
                    new_centers2d.append(centers2d)
                    new_gt_labels.append(gt_labels)
                    new_depths.append(depths)
                

                new_imgs.append(img)
                results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
                ida_mats.append(np.array(ida_mat))
                if with_depthmap:
                    new_depthmaps.append(np.array(depthmap))

        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['cam2img'] = results['intrinsics']
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in
                                range(len(results['extrinsics']))]
        
        results['img_shape'] = [img.shape for img in new_imgs]
        results['pad_shape'] = [img.shape for img in new_imgs]

        results['ida_mat'] = ida_mats    # shape N * (3, 3)

        if with_depthmap:
            results['depthmap'] = new_depthmaps

        return results

    def _bboxes_transform(self, img, bboxes, centers2d, gt_labels, depths, resize, crop, flip=False):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)

        fH, fW = img.shape[:2]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        # normalize
        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]
        # normalize

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths
    
    def offline_2d_transform(self, img, bboxes, resize, crop, flip=False):
        fH, fW = img.shape[:2]
        bboxes = np.array(bboxes).reshape(-1, 6)
        bboxes[..., :4] = bboxes[..., :4]  * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        return bboxes

    def _filter_invisible(self, img, bboxes, centers2d, gt_labels, depths):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)

        fH, fW = img.shape[:2]
        indices_maps = np.zeros((fH, fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip=False, rotate=0, depthmap=None):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # adjust depthmap (int32)
        if depthmap is not None:
            depthmap = depthmap.resize(resize_dims, resample=Image.NEAREST)
            depthmap = depthmap.crop(crop)
            if flip:
                depthmap = depthmap.transpose(method=Image.FLIP_LEFT_RIGHT)
            depthmap = depthmap.rotate(rotate, resample=Image.NEAREST)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat, depthmap

    def _sample_augmentation(self, img):
        H, W = img.shape[:2]
        fH, fW = self.data_aug_conf["final_dim"]
        resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
        crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
            flip = True
        rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        return resize, resize_dims, crop, flip, rotate

    def _sample_augmentation_f(self, img):
        H, W = img.shape[:2]
        fH, fW = W, H

        resize = np.round(((H + 50) / W), 2)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((newH - fH) / 2)
        crop_w = int((newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize, resize_dims, crop

@PIPELINES.register_module()
class AV2PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None
        
    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size == 'same2max':
            max_shape = max([img.shape for img in results['img']])[:2]
            padded_img = [mmcv.impad(img, shape=max_shape, pad_val=self.pad_val) for img in results['img']]
            with_depthmap = ('depthmap' in results.keys())
            if with_depthmap:
                results['depthmap'] = [mmcv.impad(depthmap, shape=max_shape, pad_val=self.pad_val) for depthmap in results['depthmap']]
        elif self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results['img']
            ]

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class AV2DownsampleQuantizeDepthmap():
    """
    Given downsample stride, downsample depthmap to (N, H, W), and make Quantization to return onehot format
    Input depthmap in numpy array, Output in torch array.
    """

    def __init__(self, downsample=16, grid_config=None):
        '''
        default: grid_config = [1, 150, 1]
        '''
        self.downsample = downsample
        self.grid_config = grid_config
        self.D = int(self.grid_config[1] / self.grid_config[2])    # D_max

    def __call__(self, results):
        """
        """
        depth_maps = torch.from_numpy(np.stack(results['depthmap'])) / 100.0    # (N, Hi, Wi)
        gt_depths = self._get_downsampled_gt_depth(depth_maps)  # (N, H/downsample, W/downsample)
        gt_depths_ = self._quantize_gt_depth(gt_depths)          # (N, Hd, Wd, self.D=150)
        results['depthmap'] = gt_depths_

        return results

    def _get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(N, H // self.downsample, self.downsample, W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(N, H // self.downsample, W // self.downsample)
        gt_depths = torch.where(gt_depths == 1e5, torch.zeros_like(gt_depths), gt_depths)
        return gt_depths

    def _quantize_gt_depth(self, gt_depths):
        gt_depths = (gt_depths - (self.grid_config[0] - self.grid_config[2])) / self.grid_config[2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1)[..., 1:]  # .view(-1, self.D + 1)
        return gt_depths.float()

@PIPELINES.register_module()
class AV2DownsampleQuantizeInstanceDepthmap():
    """
    Given downsample stride, downsample depthmap to (N, H, W), and make Quantization to return onehot format
    Input depthmap in numpy array, Output in torch array.
    """

    def __init__(self, downsample=8, depth_config={}):
        '''
        default: LID style
        '''
        self.downsample = downsample
        self.depth_min, self.depth_max, self.num_bins = [depth_config.get(key) for key in
                                                         ['depth_min', 'depth_max', 'num_depth_bins']]

    def __call__(self, results):
        """
        """
        # depth_maps = torch.from_numpy(np.stack(results['depthmap'])) / 100.0    # (N, Hi, Wi)
        # gt_depths = self._get_downsampled_gt_depth(depth_maps)  # (N, H/downsample, W/downsample)
        # gt_depths_ = self._quantize_gt_depth(gt_depths)          # (N, Hd, Wd, self.D=150)
        # results['depthmap'] = gt_depths_

        gt_boxes2d, gt_center_depth = results['gt_bboxes'], results['depths']   # np array, N*(M,4), N*(M)
        H, W = results['img_shape'][0][:2]
        H, W = int(H / self.downsample), int(W / self.downsample)
        depth_maps, fg_mask = self.build_target_depth_from_3dcenter_argo(gt_boxes2d, gt_center_depth, (H, W))
        depth_target = self.bin_depths(depth_maps, "LID", self.depth_min, self.depth_max, self.num_bins, target=True)
        results['ins_depthmap'] = depth_target  # (N, H, W)
        results['ins_depthmap_mask'] = fg_mask  # (N, H, W)
        return results


    def build_target_depth_from_3dcenter_argo(self, gt_boxes2d, gt_center_depth, HW):
        H, W = HW
        B = len(gt_boxes2d) # B is N indeed
        depth_maps = torch.zeros((B, H, W), dtype=torch.float)
        fg_mask = torch.zeros_like(depth_maps).bool()

        for b in range(B):
            center_depth_per_batch = torch.from_numpy(gt_center_depth[b])
            # Set box corners
            if gt_boxes2d[b].shape[0] > 0:
                gt_boxes_per_batch = torch.from_numpy(gt_boxes2d[b])
                gt_boxes_per_batch = gt_boxes_per_batch / self.downsample   # downsample is necessary
                gt_boxes_per_batch[:, :2] = torch.floor(gt_boxes_per_batch[:, :2])
                gt_boxes_per_batch[:, 2:] = torch.ceil(gt_boxes_per_batch[:, 2:])
                gt_boxes_per_batch = gt_boxes_per_batch.long()

                for n in range(gt_boxes_per_batch.shape[0]):
                    u1, v1, u2, v2 = gt_boxes_per_batch[n]
                    depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]
                    fg_mask[b, v1:v2, u1:u2] = True

        return depth_maps, fg_mask

    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        # elif mode == "SID":
        #     indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
        #               (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)

        return indices
