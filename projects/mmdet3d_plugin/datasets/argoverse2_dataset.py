import mmcv
import torch
import json
import numpy as np
from .av2_utils import yaw_to_quat
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets import DATASETS
from  mmdet3d.datasets.custom_3d import Custom3DDataset
from av2.evaluation.detection.constants import CompetitionCategories
from pathlib import Path
from .av2_eval_util import EvalNori, read_feather_remote
from .av2_utils import DetectionCfg
from .eval_recall import cal_recall_2d

LABEL_ATTR = (
    "tx_m","ty_m","tz_m","length_m","width_m","height_m","qw","qx","qy","qz",
)

@DATASETS.register_module()
class Argoverse2Dataset(Custom3DDataset):
    CLASSES = tuple(x.value for x in CompetitionCategories)
    
    def __init__(self,
                 ann_file,
                #  map_file,
                 split,
                 offline_2d_file=None,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_3d=True,
                 eval_2d=False,
                 use_offline_2d=False,
                 use_valid_flag=False,
                 **kwargs
    ):
        self.load_interval = load_interval
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs,
        )
        
        self.split = split
        self.use_valid_flag = use_valid_flag
        # map_data = mmcv.load(map_file, file_format='pkl')
        # self.map_infos = map_data[::self.load_interval]
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        # eval util
        if test_mode:
            self.eval_nori = EvalNori(data_root)
            self.val_anno_path = 's3://argo/argo_data/val_anno.feather'

        self.eval_3d = eval_3d
        self.eval_2d = eval_2d
        self.use_offline_2d = use_offline_2d
        
        if self.use_offline_2d:
            with open(offline_2d_file) as pred:
                self.offline_2d = json.load(pred)
            
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt3d_infos']['gt_names'][mask])
        else:
            gt_names = set(info['gt3d_infos']['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        # data_infos = list(sorted(data['infos'], key=lambda e: e['lidar_timestamp_ns']))
        data_infos = data['infos'][::self.load_interval]

        return data_infos
    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            scene_idx=info['scene_id'],
            # pts_filename=info['lidar_path'],
            # sweeps=info['sweeps'],
            timestamp=info['lidar_timestamp_ns'],
        )
        if self.modality['use_camera']:
            image_paths = []
            image_raw_paths = []
            lidar2img_rts = []
            distotions = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            city_SE3_ego_lidar_t = info['city_SE3_ego_lidar_t']
            # add map path

            map_path = Path(self.data_root) / 'map' / self.split / 'map-{}-{}.png'.format(info['scene_id'], info['lidar_timestamp_ns'])
            map = mmcv.imread(map_path, flag='grayscale')
            
            for _, cam_info in info['cam_infos'].items():
                if cam_info is None:
                    return None
                image_path = cam_info['fpath']
                image_paths.append(image_path)
                image_raw_paths.append(cam_info['fpath'])
                img_timestamp.append(cam_info['cam_timestamp_ns'])
                city_SE3_ego_cam_t = cam_info['city_SE3_ego_cam_t']
                ego_SE3_cam = cam_info['ego_SE3_cam']
                ego_cam_t_SE3_ego_lidar_t = city_SE3_ego_cam_t.inverse().compose(city_SE3_ego_lidar_t) #ego2glo_lidar -> glo2ego_cam
                cam_SE3_ego_cam_t = ego_SE3_cam.inverse().compose(ego_cam_t_SE3_ego_lidar_t) #ego -> cam
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = cam_SE3_ego_cam_t.rotation
                transform_matrix[:3, 3] = cam_SE3_ego_cam_t.translation

                intrinsic = cam_info['intrinsics']
                distotion = cam_info['cam_distortion']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2img = viewpad @ transform_matrix
                intrinsics.append(viewpad)
                extrinsics.append(transform_matrix)
                lidar2img_rts.append(ego2img)
                distotions.append(distotion)

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    distotion=distotions,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    city_SE3_ego=city_SE3_ego_lidar_t,
                    map_path=map_path, # add map path
                    map_ori=np.array(map).astype(np.float64)
                    # map=self.map_infos[index]['map_vector'],
                ))
            
        if self.use_offline_2d:
            offline_2ds = []
            for image_raw_path in image_raw_paths:
                offline_2ds.append(self.offline_2d[str(image_raw_path)])
            input_dict['offline_2d'] = offline_2ds

        if not self.test_mode:
            annos = self.get_ann_info(index, input_dict)
            annos.update( 
                dict(
                    bboxes=info['gt2d_infos']['gt_2dbboxes'],
                    labels=info['gt2d_infos']['gt_2dlabels'],
                    centers2d=info['gt2d_infos']['centers2d'],
                    depths=info['gt2d_infos']['depths'],)
            )
            input_dict['ann_info'] = annos
        return input_dict
    
    def get_ann_info(self, index, input_dict):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]['gt3d_infos']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_interior_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = np.array(info['gt_names'])[mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, #xyzlwh+yaw
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results
        
    
    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 jsonfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 eval_range_m=None):
        # from av2.evaluation.detection.utils import DetectionCfg
        # from av2.evaluation.detection.eval import evaluate
        # from av2.utils.io import read_all_annotations, read_feather

        dts, dts_2d_dict= self.format_results(results, jsonfile_prefix, submission_prefix)
        # val_anno_path = osp.join(self.data_root, 'val_anno.feather')
        if self.eval_3d and self.eval_2d:
            gts = read_feather_remote(self.val_anno_path)
            gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

            valid_uuids_gts = gts.index.tolist()
            valid_uuids_dts = dts.index.tolist()
            valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
            gts = gts.loc[list(valid_uuids)].sort_index()

            categories = set(x.value for x in CompetitionCategories)
            categories &= set(gts["category"].unique().tolist())  # 交集
            split_dir = Path(self.data_root) / self.split
            cfg = DetectionCfg(
                dataset_dir=split_dir,
                categories=tuple(sorted(categories)),
                eval_range_m=[0.0, 150.0] if eval_range_m is None else eval_range_m,
                eval_only_roi_instances=True,
            )
            
            # eval_dts, eval_gts, metrics = evaluate(dts.reset_index(), gts.reset_index(), cfg)
            eval_dts, eval_gts, metrics, recall_3d = self.eval_nori.evaluate(dts.reset_index(), gts.reset_index(), cfg)

            valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
            print(metrics.loc[valid_categories])
            print(recall_3d)

            cal_recall_2d(dts_2d_dict)

            ap_dict = {}
            for index, row in metrics.iterrows():
                # for k, v in row.to_dict().items():
                #     name = (
                #         "hp/CDS"
                #         if k == "CDS" and (index == "AVERAGE_METRICS")
                #         else f"hp/{k}/{index}"
                #     )
                #     ap_dict[name] = v
                ap_dict[index] = row.to_json()

        elif self.eval_3d:
            gts = read_feather_remote(self.val_anno_path)
            gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

            valid_uuids_gts = gts.index.tolist()
            valid_uuids_dts = dts.index.tolist()
            valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
            gts = gts.loc[list(valid_uuids)].sort_index()

            categories = set(x.value for x in CompetitionCategories)
            categories &= set(gts["category"].unique().tolist())  # 交集
            split_dir = Path(self.data_root) / self.split
            cfg = DetectionCfg(
                dataset_dir=split_dir,
                categories=tuple(sorted(categories)),
                eval_range_m=[0.0, 150.0] if eval_range_m is None else eval_range_m,
                eval_only_roi_instances=True,
            )
            eval_dts, eval_gts, metrics, recall_3d = self.eval_nori.evaluate(dts.reset_index(), gts.reset_index(), cfg)
            valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
            print(metrics.loc[valid_categories])
            print(recall_3d)
            ap_dict = {}
            for index, row in metrics.iterrows():
                ap_dict[index] = row.to_json()
        
        elif self.eval_2d:
            cal_recall_2d(dts_2d_dict)
            ap_dict = {}
        
        else:
            ap_dict = {}

        return ap_dict
        
    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       ):
        """Format the results to .feather file with argo2 format.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        import pandas as pd

        assert len(self.data_infos) == len(outputs)
        num_samples = len(outputs)
        print('\nGot {} samples'.format(num_samples))
        
        serialized_dts_list = []
        dts_2d_dict = {}
        if self.eval_3d and self.eval_2d:
            print('\nConvert predictions to Argoverse 2 format')
            for i in mmcv.track_iter_progress(range(num_samples)):
                out_i = outputs[i]
                if 'pts_bbox' in out_i:
                    out_i = out_i['pts_bbox'] # for MVX-style detector
                log_id = self.data_infos[i]['scene_id']
                ts = self.data_infos[i]['lidar_timestamp_ns']
                token = str(log_id) + str(ts)
                dts_2d_dict[token] = outputs[i]['bbox_roi']
                cat_id = out_i['labels_3d'].numpy().tolist()
                category = [self.CLASSES[i].upper() for i in cat_id]
                serialized_dts = pd.DataFrame(
                    self.box_to_av2(out_i['boxes_3d']).numpy(), columns=list(LABEL_ATTR)
                )
                serialized_dts["score"] = out_i['scores_3d'].numpy()
                serialized_dts["log_id"] = log_id
                serialized_dts["timestamp_ns"] = int(ts)
                serialized_dts["category"] = category
                serialized_dts_list.append(serialized_dts)
            
            dts = (
                pd.concat(serialized_dts_list)
                .set_index(["log_id", "timestamp_ns"])
                .sort_index()
            )

            dts = dts.sort_values("score", ascending=False).reset_index()

            if pklfile_prefix is not None:
                mmcv.mkdir_or_exist(pklfile_prefix)
                if not pklfile_prefix.endswith(('.feather')):
                    pklfile_prefix = f'{pklfile_prefix}.feather'
                dts.to_feather(pklfile_prefix)
                print(f'Result is saved to {pklfile_prefix}.')

            dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()
        
        elif self.eval_3d:
            print('\nConvert predictions to Argoverse 2 format')
            for i in mmcv.track_iter_progress(range(num_samples)):
                out_i = outputs[i]
                if 'pts_bbox' in out_i:
                    out_i = out_i['pts_bbox'] # for MVX-style detector
                log_id = self.data_infos[i]['scene_id']
                ts = self.data_infos[i]['lidar_timestamp_ns']
                cat_id = out_i['labels_3d'].numpy().tolist()
                category = [self.CLASSES[i].upper() for i in cat_id]
                serialized_dts = pd.DataFrame(
                    self.box_to_av2(out_i['boxes_3d']).numpy(), columns=list(LABEL_ATTR)
                )
                serialized_dts["score"] = out_i['scores_3d'].numpy()
                serialized_dts["log_id"] = log_id
                serialized_dts["timestamp_ns"] = int(ts)
                serialized_dts["category"] = category
                serialized_dts_list.append(serialized_dts)
            
            dts = (
                pd.concat(serialized_dts_list)
                .set_index(["log_id", "timestamp_ns"])
                .sort_index()
            )

            dts = dts.sort_values("score", ascending=False).reset_index()

            if pklfile_prefix is not None:
                mmcv.mkdir_or_exist(pklfile_prefix)
                if not pklfile_prefix.endswith(('.feather')):
                    pklfile_prefix = f'{pklfile_prefix}.feather'
                dts.to_feather(pklfile_prefix)
                print(f'Result is saved to {pklfile_prefix}.')

            dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()

        elif self.eval_2d:
            for i in mmcv.track_iter_progress(range(num_samples)):
                log_id = self.data_infos[i]['scene_id']
                ts = self.data_infos[i]['lidar_timestamp_ns']
                token = str(log_id) + str(ts)
                dts_2d_dict[token] = outputs[i]['bbox_roi']
            dts = None
        
        else:
            dts = None

        return dts, dts_2d_dict
    
    def box_to_av2(self, boxes):
        cnt_xyz = boxes.gravity_center
        # lwh = boxes.tensor[:, [4, 3, 5]]
        lwh = boxes.tensor[:, [3, 4, 5]]

        yaw = boxes.tensor[:, 6]

        # yaw = -yaw - 0.5 * np.pi
        # while (yaw < -np.pi).any():
        #     yaw[yaw < -np.pi] += 2 * np.pi
        # while (yaw > np.pi).any():
        #     yaw[yaw > np.pi] -= 2 * np.pi

        quat = yaw_to_quat(yaw)
        argo_cuboid = torch.cat([cnt_xyz, lwh, quat], dim=1)
        return argo_cuboid






