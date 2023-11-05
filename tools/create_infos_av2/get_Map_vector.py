# save HD Map in padding points format

from pathlib import Path
import numpy as np
from av2.utils.io import  read_city_SE3_ego
import mmcv
from av2.utils import io
from av2.map.drivable_area import DrivableArea
import copy
from typing import Final, List
import av2.utils.dilation_utils as dilation_utils
import av2.utils.raster as raster_utils
from av2.geometry.sim2 import Sim2
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import interpolate
import os
import torch
# 1 meter resolution is insufficient for the online-generated drivable area and ROI raster grids
# these grids can be generated at an arbitrary resolution, from vector (polygon) objects.
ONLINE_RASTER_RESOLUTION_M: Final[float] = 0.1  # 10 cm resolution
ONLINE_RASTER_RESOLUTION_SCALE: Final[float] = 1 / ONLINE_RASTER_RESOLUTION_M

ROI_ISOCONTOUR_M: Final[float] = 5.0  # in meters
ROI_ISOCONTOUR_GRID: Final[float] = ROI_ISOCONTOUR_M * ONLINE_RASTER_RESOLUTION_SCALE
BOUNDRARY = [-152.4, -152.4, 152.4, 152.4]

dataroot = Path("/data/av2")

def get_map(split):
    
    pkl_path = dataroot / "av2_{}_infos.pkl".format(split)
    data = mmcv.load(pkl_path, file_format='pkl')
    data_infos = data['infos']
    data_dir = dataroot / split

    drivable_areas_points = []
    for i in tqdm(range(len(data_infos))):
        info = data_infos[i]
        ts = info["lidar_timestamp_ns"]
        log_id = info["scene_id"]
        log_dir = Path(data_dir) / log_id
        
        log_map_dirpath = log_dir / "map"
        # Load vector map data from JSON file
        vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
        if not len(vector_data_fnames) == 1:
            raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
        vector_data_fname = vector_data_fnames[0]
        vector_data_json_path = log_map_dirpath / vector_data_fname
        vector_data = io.read_json_file(vector_data_json_path)
        vector_drivable_areas = {da["id"]: DrivableArea.from_dict(da) for da in vector_data["drivable_areas"].values()}

        drivable_areas: List[DrivableArea] = list(vector_drivable_areas.values())
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)
        city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[ts]

        drivable_points = [city_SE3_ego_lidar_t.inverse().transform_from(da.xyz[:, :3]) for da in drivable_areas]
        
        drivable_areas_point = {}
        da_polygons = []
        drivable_areas_num = 0
        for da_polygon_city in drivable_points:
            is_valid = np.where(np.logical_and(abs(da_polygon_city[:, 0]) < 152.4, abs(da_polygon_city[:, 1]) < 152.4))
            if is_valid[0].shape[0] < 4:
                continue
            da_polygon_city_valid = da_polygon_city[is_valid]
            tck, _ = interpolate.splprep([da_polygon_city_valid[:, 0], da_polygon_city_valid[:, 1]], s=0, per=True)
            xi, yi = interpolate.splev(np.linspace(0, 1, 80), tck)
            polygon_city_valid = torch.stack((torch.from_numpy(xi), torch.from_numpy(yi)), dim=-1)
            da_polygons.append(polygon_city_valid.numpy())
            drivable_areas_num += 1
        
        da_polygons = np.array(da_polygons)
        map_vector = np.zeros([32, 80, 2])
        map_vector[:da_polygons.shape[0]] = da_polygons
        if drivable_areas_num > 32:
            print('-------------------')
        drivable_areas_point['drivable_areas_num'] = drivable_areas_num
        drivable_areas_point['map_vector'] = map_vector
        
        drivable_areas_points.append(drivable_areas_point)
    print('{}_sample:{}'.format(split, len(drivable_areas_points))) 
    save_path = dataroot / 'av2_drivable_areas_{}.pkl'.format(split)
    mmcv.dump(drivable_areas_points, save_path)

if __name__ == '__main__':
    splits = ["val", "test", "train"]
    for split in splits:
        get_map(split)