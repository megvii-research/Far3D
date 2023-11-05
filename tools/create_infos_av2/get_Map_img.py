# save HD Map in image format

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
import os
# 1 meter resolution is insufficient for the online-generated drivable area and ROI raster grids
# these grids can be generated at an arbitrary resolution, from vector (polygon) objects.
ONLINE_RASTER_RESOLUTION_M: Final[float] = 0.1  # 10 cm resolution
ONLINE_RASTER_RESOLUTION_SCALE: Final[float] = 1 / ONLINE_RASTER_RESOLUTION_M

ROI_ISOCONTOUR_M: Final[float] = 1.0  # in meters
ROI_ISOCONTOUR_GRID: Final[float] = ROI_ISOCONTOUR_M * ONLINE_RASTER_RESOLUTION_SCALE
BOUNDRARY = [-152.4, -152.4, 152.4, 152.4]

dataroot = Path("/data/av2")

def get_map(split):
    
    pkl_path = dataroot / "av2_{}_infos.pkl".format(split)
    data = mmcv.load(pkl_path, file_format='pkl')
    data_infos = data['infos']
    data_dir = dataroot / split

    for info in tqdm(data_infos):
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

        array_s_city = ONLINE_RASTER_RESOLUTION_SCALE
        img_h = int((BOUNDRARY[2] - BOUNDRARY[0]) * array_s_city)
        img_w = int((BOUNDRARY[3] - BOUNDRARY[1]) * array_s_city)

        # scale determines the resolution of the raster DA layer.
        array_Sim2_city = Sim2(R=np.eye(2), t=np.array([-BOUNDRARY[0], -BOUNDRARY[1]]), s=array_s_city)

        da_polygons_img = []
        for da_polygon_city in drivable_points:
            da_polygon_img = array_Sim2_city.transform_from(da_polygon_city[:, :2])
            da_polygon_img = np.round(da_polygon_img).astype(np.int32)
            da_polygons_img.append(da_polygon_img)

        da_mask = raster_utils.get_mask_from_polygons(da_polygons_img, img_h, img_w)
        roi_mat_init = copy.deepcopy(da_mask).astype(np.uint8)
        roi_mask = np.flipud(dilation_utils.dilate_by_l2(roi_mat_init, dilation_thresh=ROI_ISOCONTOUR_GRID))

        im = Image.fromarray(roi_mask)
        dir_path = dataroot / "map" /split
        os.makedirs(dir_path, exist_ok=True)
        save_path = dir_path / 'map-{}-{}.png'.format(log_id, ts)
        im.save(save_path)
    

if __name__ == '__main__':
    splits = ["train", "val", "test"]
    for split in splits:
        get_map(split)