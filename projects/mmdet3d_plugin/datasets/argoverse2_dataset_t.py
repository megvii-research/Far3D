import torch
import numpy as np
from mmdet.datasets import DATASETS
from av2.evaluation.detection.constants import CompetitionCategories
from pathlib import Path
from .argoverse2_dataset import Argoverse2Dataset
import math
from mmcv.parallel import DataContainer as DC
import random

LABEL_ATTR = (
    "tx_m","ty_m","tz_m","length_m","width_m","height_m","qw","qx","qy","qz",
)

@DATASETS.register_module()
class Argoverse2DatasetT(Argoverse2Dataset):
    CLASSES = tuple(x.value for x in CompetitionCategories)
    
    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, interval_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode

        if interval_test:
            data_infos = self.data_infos
            s1, s2, s3, s4, s5 = data_infos[::5], data_infos[1::5], data_infos[2::5], data_infos[3::5], data_infos[4::5]
            data_infos = s1 + s2 + s3 + s4 + s5 
            self.data_infos = data_infos
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.


    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []
        scene_id = None

        curr_sequence = -1
        for idx in range(len(self.data_infos)):
            if self.data_infos[idx]['scene_id'] != scene_id:
                # Not first frame and # of sweeps is 0 -> new sequence
                scene_id = self.data_infos[idx]['scene_id']
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                # assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                
                self.flag = np.array(new_flags, dtype=np.int64)


    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            # print(len(np.bincount(self.flag)))

            if input_dict is None:
                return None
            
            if not self.seq_mode:
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def union2one(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        city_SE3_ego = info['city_SE3_ego_lidar_t']
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = city_SE3_ego.rotation
        transform_matrix[:3, 3] = city_SE3_ego.translation

        ego_pose =  transform_matrix

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        pts_filename = Path(self.split)/ info['scene_id']  / 'sensors' /  'lidar' / f"{info['lidar_timestamp_ns']}.feather"
        input_dict = dict(
            pts_filename=pts_filename,
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            scene_token=info['scene_id'],
            timestamp=index,
            lidar_timestamp=info['lidar_timestamp_ns'],
        )

        if self.modality['use_camera']:
            image_paths = []
            image_raw_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            city_SE3_ego_lidar_t = info['city_SE3_ego_lidar_t']
            for cam_type, cam_info in info['cam_infos'].items():
                if cam_info is None:
                    return None
                img_timestamp.append(cam_info['cam_timestamp_ns']/ 1e9)
                image_path = self.data_root / cam_info['fpath']
                image_paths.append(image_path)
                image_raw_paths.append(cam_info['fpath'])
                # obtain lidar to image transformation matrix
                city_SE3_ego_cam_t = cam_info['city_SE3_ego_cam_t']
                ego_SE3_cam = cam_info['ego_SE3_cam']
                ego_cam_t_SE3_ego_lidar_t = city_SE3_ego_cam_t.inverse().compose(city_SE3_ego_lidar_t) #ego2glo_lidar -> glo2ego_cam
                cam_SE3_ego_cam_t = ego_SE3_cam.inverse().compose(ego_cam_t_SE3_ego_lidar_t) #ego -> cam
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = cam_SE3_ego_cam_t.rotation
                transform_matrix[:3, 3] = cam_SE3_ego_cam_t.translation

                intrinsic = cam_info['intrinsics']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ transform_matrix)
                intrinsics.append(viewpad)
                extrinsics.append(transform_matrix)
                lidar2img_rts.append(lidar2img_rt)
                
            if not self.test_mode:
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None
            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index, input_dict)
            annos.update( 
                dict(
                    bboxes=info['gt2d_infos']['gt_2dbboxes'],
                    labels=info['gt2d_infos']['gt_2dlabels'],
                    centers2d=info['gt2d_infos']['centers2d'],
                    depths=info['gt2d_infos']['depths'],
                    bboxes_ignore=None)
            )
            input_dict['ann_info'] = annos
            
        return input_dict


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.
        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag != -1)[0]
        return np.random.choice(pool)

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix





