# create depthmap

from pathlib import Path
from typing import Optional
import numpy as np
import av2.utils.dense_grid_interpolation as dense_grid_interpolation
import av2.utils.io as io_utils
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.utils.io import  read_city_SE3_ego
from av2.utils.typing import NDArrayFloat
from os import path as osp
from pathlib import Path
import numpy as np
from av2.utils.io import  read_city_SE3_ego
import mmcv
from av2.datasets.sensor.constants import RingCameras
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
CAMERAS = tuple(x.value for x in RingCameras)

dataroot = Path("/data/av2")

def get_depthmap(split):
    
    pkl_path = dataroot / "av2_{}_infos.pkl".format(split)
    data = mmcv.load(pkl_path, file_format='pkl')
    data_infos = data['infos']
    data_dir = dataroot / split
    
    saved_files =[]
    for info in tqdm(data_infos):
        ts = info["lidar_timestamp_ns"]
        log_id = info["scene_id"]
        
        # depthmaps = []
        for cam_name in CAMERAS:
            if info['cam_infos'][cam_name] is None:
                print("No corresponding ring image")
                continue
            depthmap = get_depth_map_from_lidar(
                data_dir=data_dir,
                cam_name=cam_name,
                log_id=log_id,
                cam_timestamp_ns=info['cam_infos'][cam_name]['cam_timestamp_ns'],
                lidar_timestamp_ns=ts,
                interp_depth_map=False,
            )
            # depthmaps.append(depthmap)

            im = Image.fromarray((depthmap * 100).astype(np.int32))
            split_path = dataroot / "depth" / split
            dir_path = split_path / str(log_id) / str(ts)
            os.makedirs(dir_path, exist_ok=True)
            save_path = str(dir_path) + '/%s.png' % (cam_name)
            im.save(save_path)
        
            saved_files.append(str(split_path).split('/')[-1] + '/%s/%s/%s.tiff' % (log_id, ts, cam_name))
    
    record_dir_path = dataroot / "depth" 
    print('{}_sample:{}'.format(split, len(saved_files)))
    record_path = osp.join(record_dir_path, '{}.pkl'.format(split))
    mmcv.dump(saved_files, record_path)

def get_depth_map_from_lidar(
        data_dir:Path,
        cam_name: str,  
        log_id: str,
        cam_timestamp_ns: int,
        lidar_timestamp_ns: int,
        interp_depth_map: bool = True,
    ) -> Optional[NDArrayFloat]:
    
        lidar_fname = f"{lidar_timestamp_ns}.feather"
        lidar_fpath = data_dir / log_id / "sensors" / "lidar" / lidar_fname
        lidar_points = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
        log_dir = data_dir / log_id
        pinhole_camera = PinholeCamera.from_feather(log_dir=log_dir, cam_name=cam_name)
        height_px, width_px = pinhole_camera.height_px, pinhole_camera.width_px

        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)
        city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[lidar_timestamp_ns]
        city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_timestamp_ns]

        uv, points_cam, is_valid_points = pinhole_camera.project_ego_to_img_motion_compensated(
            points_lidar_time=lidar_points,
            city_SE3_ego_cam_t=city_SE3_ego_cam_t,
            city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
        )

        if uv is None or points_cam is None:
            # poses were missing for either the camera or lidar timestamp
            return None
        if is_valid_points is None or is_valid_points.sum() == 0:
            return None

        u = np.round(uv[:, 0][is_valid_points]).astype(np.int32)
        v = np.round(uv[:, 1][is_valid_points]).astype(np.int32)
        z = points_cam[:, 2][is_valid_points]

        depth_map: NDArrayFloat = np.zeros((height_px, width_px), dtype=np.float32)

        # form depth map from LiDAR
        if interp_depth_map:
            if u.max() > pinhole_camera.width_px or v.max() > pinhole_camera.height_px:
                raise RuntimeError("Regular grid interpolation will fail due to out-of-bound inputs.")

            depth_map = dense_grid_interpolation.interp_dense_grid_from_sparse(
                grid_img=depth_map,
                points=np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)]),
                values=z,
                grid_h=height_px,
                grid_w=width_px,
                interp_method="linear",
            ).astype(float)
        else:
            depth_map[v, u] = z

        return depth_map
    
    
if __name__ == '__main__':
    splits = ["val", "test", "train"]
    for split in splits:
        get_depthmap(split)