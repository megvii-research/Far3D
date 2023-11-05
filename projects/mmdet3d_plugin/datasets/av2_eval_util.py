'''
Overwrite some functions for adapting to our data loading scheme.
'''

import pickle
import refile
import numpy as np
import nori2 as nori
import pandas
import pyarrow
import pyarrow.feather as feather

import logging
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd

from av2.evaluation.detection.constants import NUM_DECIMALS, MetricNames, TruePositiveErrorNames
from av2.evaluation.detection.utils import (
    # DetectionCfg,
    # accumulate,
    compute_average_precision,
    groupby,
    load_mapped_avm_and_egoposes,
)
from .av2_utils import (
    DetectionCfg,
    accumulate,
)
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.io import TimestampedCitySE3EgoPoses
from av2.utils.typing import NDArrayBool, NDArrayFloat
import av2.geometry.geometry as geometry_utils

# from av2.evaluation.detection.eval import summarize_metrics
from .summarize_metrics_av2 import summarize_metrics
from typing import Dict, List, Optional, Tuple, Union, Final, Any
from joblib import Parallel, delayed
from pathlib import Path
from upath import UPath
from io import BytesIO

from dataclasses import dataclass
import json
from av2.map.drivable_area import DrivableArea
from av2.map.lane_segment import LaneSegment
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.map.map_api import DrivableAreaMapLayer, RoiMapLayer, GroundHeightLayer
from av2.geometry.sim2 import Sim2

warnings.filterwarnings("ignore", module="google")

TP_ERROR_COLUMNS: Final[Tuple[str, ...]] = tuple(x.value for x in TruePositiveErrorNames)
DTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("score",)
GTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("num_interior_pts",)
UUID_COLUMN_NAMES: Final[Tuple[str, ...]] = (
    "log_id",
    "timestamp_ns",
    "category",
)

logger = logging.getLogger(__name__)

def read_feather_remote(path: Union[Path, UPath], columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """Read Apache Feather data from a .feather file.
    AV2 uses .feather to serialize much of its data. This function handles the deserialization
    process and returns a `pandas` DataFrame with rows corresponding to the records and the
    columns corresponding to the record attributes.
    Args:
        path: Source data file (e.g., 'lidar.feather', 'calibration.feather', etc.)
        columns: Tuple of columns to load for the given record. Defaults to None.
    Returns:
        (N,len(columns)) Apache Feather data represented as a `pandas` DataFrame.
    """
    # with path.open("rb") as f:
    with refile.smart_open(str(path), "rb") as f:
        data: pd.DataFrame = feather.read_feather(f, columns=columns)
    return data

class EvalNori():
    def __init__(self,
                 data_root='data/av2/',
                 SE3_ego_pkl_name = 's3://argo/nori/0210/argoverse2_city_SE3_egovehicle_0.pkl',
                 SE3_ego_nori='s3://argo/nori/0210/argoverse2_city_SE3_egovehicle.nori',
                 ):
        self.data_root = data_root
        self.fetcher = nori.Fetcher()
        with refile.smart_open(SE3_ego_pkl_name, "rb") as f:
            self.name2nori = dict(pickle.load(f))

    def _get_feather_byte(self, filename):
        nori_id = self.name2nori.get(str(filename), -1)
        feather_byte = self.fetcher.get(nori_id)
        return feather_byte

    def convert_byte_to_dataframe(self, feather_byte):
        reader = pyarrow.BufferReader(feather_byte)
        data: pandas.DataFrame = feather.read_feather(reader)
        return data



    def read_city_SE3_ego(self, log_dir: Union[Path, UPath]) -> TimestampedCitySE3EgoPoses:
        """Read the egovehicle poses in the city reference frame.
        The egovehicle city pose defines an SE3 transformation from the egovehicle reference frame to the city ref. frame.
        Mathematically we define this transformation as: $$city_SE3_ego$$.
        In other words, when this transformation is applied to a set of points in the egovehicle reference frame, they
        will be transformed to the city reference frame.
        Example (1).
            points_city = city_SE3_ego(points_ego) applying the SE3 transformation to points in the egovehicle ref. frame.
        Example (2).
            ego_SE3_city = city_SE3_ego^{-1} take the inverse of the SE3 transformation.
            points_ego = ego_SE3_city(points_city) applying the SE3 transformation to points in the city ref. frame.
        Extrinsics:
            timestamp_ns: Egovehicle nanosecond timestamp.
            qw: scalar component of a quaternion.
            qx: X-axis coefficient of a quaternion.
            qy: Y-axis coefficient of a quaternion.
            qz: Z-axis coefficient of a quaternion.
            tx_m: X-axis translation component.
            ty_m: Y-axis translation component.
            tz_m: Z-axis translation component.
        Args:
            log_dir: Path to the log directory.
        Returns:
            Mapping from egovehicle time (in nanoseconds) to egovehicle pose in the city reference frame.
        """

        # original code
        # city_SE3_ego_path = log_dir /
        # city_SE3_ego = read_feather(city_SE3_ego_path)

        # TODO modify: load from nori, need debug
        # such as 'train/00a6ffc1-6ce9-3bc3-a060-6006e9893ala/city_SE3_egovehicle.feather' TODO
        city_SE3_ego_path = str(log_dir).split(self.data_root)[1] + "/city_SE3_egovehicle.feather"
        city_SE3_ego = self.convert_byte_to_dataframe(self._get_feather_byte(city_SE3_ego_path))


        quat_wxyz = city_SE3_ego.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy()
        translation_xyz_m = city_SE3_ego.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()
        timestamps_ns = city_SE3_ego["timestamp_ns"].to_numpy()

        rotation = geometry_utils.quat_to_mat(quat_wxyz)
        timestamp_city_SE3_ego_dict: TimestampedCitySE3EgoPoses = {
            ts: SE3(rotation=rotation[i], translation=translation_xyz_m[i]) for i, ts in enumerate(timestamps_ns)
        }
        return timestamp_city_SE3_ego_dict

    def load_mapped_avm_and_egoposes(self, log_ids: List[str], dataset_dir: Union[Path, UPath]
    ) -> Tuple[Dict[str, ArgoverseStaticMap], Dict[str, TimestampedCitySE3EgoPoses]]:
        """Load the maps and egoposes for each log in the dataset directory.
        Args:
            log_ids: List of the log_ids.
            dataset_dir: Directory to the dataset.
        Returns:
            A tuple of mappings from log id to maps and timestamped-egoposes, respectively.
        Raises:
            RuntimeError: If the process for loading maps and timestamped egoposes fails.
        """
        log_id_to_timestamped_poses = {log_id: self.read_city_SE3_ego(dataset_dir / log_id) for log_id in log_ids}
        avms: Optional[List[ArgoverseStaticMap]] = Parallel(n_jobs=-1, backend="threading")(
            delayed(ArgoverseStaticMapRemote.from_map_dir_remote)(dataset_dir / log_id / "map", build_raster=True,
                data_root=self.data_root) for log_id in log_ids)

        if avms is None:
            raise RuntimeError("Map and egopose loading has failed!")
        log_id_to_avm = {log_ids[i]: avm for i, avm in enumerate(avms)}
        return log_id_to_avm, log_id_to_timestamped_poses

    def evaluate(self,
            dts: pd.DataFrame,
            gts: pd.DataFrame,
            cfg: DetectionCfg,
            n_jobs: int = 8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Evaluate a set of detections against the ground truth annotations.
        Each sweep is processed independently, computing assignment between detections and ground truth annotations.
        Args:
            dts: (N,14) Table of detections.
            gts: (M,15) Table of ground truth annotations.
            cfg: Detection configuration.
            n_jobs: Number of jobs running concurrently during evaluation.
        Returns:
            (C+1,K) Table of evaluation metrics where C is the number of classes. Plus a row for their means.
            K refers to the number of evaluation metrics.
        Raises:
            RuntimeError: If accumulation fails.
            ValueError: If ROI pruning is enabled but a dataset directory is not specified.
        """
        if cfg.eval_only_roi_instances and cfg.dataset_dir is None:
            raise ValueError(
                "ROI pruning has been enabled, but the dataset directory has not be specified. "
                "Please set `dataset_directory` to the split root, e.g. av2/sensor/val."
            )

        # Sort both the detections and annotations by lexicographic order for grouping.
        dts = dts.sort_values(list(UUID_COLUMN_NAMES))
        gts = gts.sort_values(list(UUID_COLUMN_NAMES))

        dts_npy: NDArrayFloat = dts[list(DTS_COLUMN_NAMES)].to_numpy()
        gts_npy: NDArrayFloat = gts[list(GTS_COLUMN_NAMES)].to_numpy()

        dts_uuids: List[str] = dts[list(UUID_COLUMN_NAMES)].to_numpy().tolist()
        gts_uuids: List[str] = gts[list(UUID_COLUMN_NAMES)].to_numpy().tolist()

        # We merge the unique identifier -- the tuple of ("log_id", "timestamp_ns", "category")
        # into a single string to optimize the subsequent grouping operation.
        # `groupby_mapping` produces a mapping from the uuid to the group of detections / annotations
        # which fall into that group.
        uuid_to_dts = groupby([":".join(map(str, x)) for x in dts_uuids], dts_npy)
        uuid_to_gts = groupby([":".join(map(str, x)) for x in gts_uuids], gts_npy)

        log_id_to_avm: Optional[Dict[str, ArgoverseStaticMap]] = None
        log_id_to_timestamped_poses: Optional[Dict[str, TimestampedCitySE3EgoPoses]] = None

        # Load maps and egoposes if roi-pruning is enabled.
        if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
            logger.info("Loading maps and egoposes ...")
            log_ids: List[str] = gts.loc[:, "log_id"].unique().tolist()
            log_id_to_avm, log_id_to_timestamped_poses = self.load_mapped_avm_and_egoposes(log_ids, cfg.dataset_dir)

        args_list: List[
            Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]] = []
        uuids = sorted(uuid_to_dts.keys() | uuid_to_gts.keys())
        for uuid in uuids:
            log_id, timestamp_ns, _ = uuid.split(":")
            args: Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]

            sweep_dts: NDArrayFloat = np.zeros((0, 10))
            sweep_gts: NDArrayFloat = np.zeros((0, 10))
            if uuid in uuid_to_dts:
                sweep_dts = uuid_to_dts[uuid]
            if uuid in uuid_to_gts:
                sweep_gts = uuid_to_gts[uuid]

            args = sweep_dts, sweep_gts, cfg, None, None
            if log_id_to_avm is not None and log_id_to_timestamped_poses is not None:
                avm = log_id_to_avm[log_id]
                city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]
                args = sweep_dts, sweep_gts, cfg, avm, city_SE3_ego
            args_list.append(args)

        logger.info("Starting evaluation ...")
        with mp.get_context("spawn").Pool(processes=n_jobs) as p:
            outputs: Optional[List[Tuple[NDArrayFloat, NDArrayFloat]]] = p.starmap(accumulate, args_list)

        if outputs is None:
            raise RuntimeError(
                "Accumulation has failed! Please check the integrity of your detections and annotations.")
        dts_list, gts_list = zip(*outputs)

        METRIC_COLUMN_NAMES = cfg.affinity_thresholds_m + TP_ERROR_COLUMNS + ("is_evaluated",)
        dts_metrics: NDArrayFloat = np.concatenate(dts_list)
        gts_metrics: NDArrayFloat = np.concatenate(gts_list)
        dts.loc[:, METRIC_COLUMN_NAMES] = dts_metrics
        gts.loc[:, METRIC_COLUMN_NAMES] = gts_metrics

        # Compute summary metrics.
        metrics, recall3d = summarize_metrics(dts, gts, cfg)
        metrics.loc["AVERAGE_METRICS"] = metrics.mean()
        recall3d.loc["AVERAGE_METRICS"] = recall3d.mean()
        metrics = metrics.round(NUM_DECIMALS)
        recall3d = recall3d.round(NUM_DECIMALS)
        return dts, gts, metrics, recall3d


@dataclass
class ArgoverseStaticMapRemote(ArgoverseStaticMap):
    """API to interact with a local map for a single log (within a single city).
    """

    @classmethod
    def from_json_remote(cls, static_map_path_s3, static_map_path: Union[Path, UPath]) -> ArgoverseStaticMap:
        """Instantiate an Argoverse static map object (without raster data) from a JSON file containing map data.
        Args:
            static_map_path: Path to the JSON file containing map data. The file name must match
                the following pattern: "log_map_archive_{log_id}.json".
        Returns:
            An Argoverse HD map.
        """
        log_id = static_map_path.stem.split("log_map_archive_")[1]
        # vector_data = io.read_json_file(static_map_path)
        with refile.smart_open(static_map_path_s3, "rb") as f:
            vector_data: Dict[str, Any] = json.load(f)

        vector_drivable_areas = {da["id"]: DrivableArea.from_dict(da) for da in vector_data["drivable_areas"].values()}
        vector_lane_segments = {ls["id"]: LaneSegment.from_dict(ls) for ls in vector_data["lane_segments"].values()}

        if "pedestrian_crossings" not in vector_data:
            logger.error("Missing Pedestrian crossings!")
            vector_pedestrian_crossings = {}
        else:
            vector_pedestrian_crossings = {
                pc["id"]: PedestrianCrossing.from_dict(pc) for pc in vector_data["pedestrian_crossings"].values()
            }

        return cls(
            log_id=log_id,
            vector_drivable_areas=vector_drivable_areas,
            vector_lane_segments=vector_lane_segments,
            vector_pedestrian_crossings=vector_pedestrian_crossings,
            raster_drivable_area_layer=None,
            raster_roi_layer=None,
            raster_ground_height_layer=None,
        )

    @classmethod
    def from_map_dir_remote(cls, log_map_dirpath: Union[Path, UPath], build_raster: bool = False, data_root='') -> ArgoverseStaticMap:
        """Instantiate an Argoverse map object from data stored within a map data directory.
        Note: the ground height surface file and associated coordinate mapping is not provided for the
        2.0 Motion Forecasting dataset, so `build_raster` defaults to False. If raster functionality is
        desired, users should pass `build_raster` to True (e.g. for the Sensor Datasets and Map Change Datasets).
        Args:
            log_map_dirpath: Path to directory containing scenario-specific map data,
                JSON file must follow this schema: "log_map_archive_{log_id}.json".
            build_raster: Whether to rasterize drivable areas, compute region of interest BEV binary segmentation,
                and to load raster ground height from disk (when available).
        Returns:
            The HD map.
        Raises:
            RuntimeError: If the vector map data JSON file is missing.
        """
        log_map_dirpath_s3 = str(log_map_dirpath).replace(data_root, 's3://argo/argo_data/')
        vector_data_fnames = sorted(refile.smart_glob(log_map_dirpath_s3 + "/log_map_archive_*.json"))
        # "s3://argo/argo_data/val/02678d04-cc9f-3148-9f95-1ba66347dff9/map/log_map_archive_*.json"
        # Load vector map data from JSON file
        # vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
        if not len(vector_data_fnames) == 1:
            raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
        vector_data_fname = vector_data_fnames[0]

        vector_data_json_path = vector_data_fname

        static_map = cls.from_json_remote(vector_data_json_path, Path(vector_data_json_path))
        static_map.log_id = log_map_dirpath.parent.stem

        # Avoid file I/O and polygon rasterization when not needed
        if build_raster:
            drivable_areas: List[DrivableArea] = list(static_map.vector_drivable_areas.values())
            static_map.raster_drivable_area_layer = DrivableAreaMapLayer.from_vector_data(drivable_areas=drivable_areas)
            static_map.raster_roi_layer = RoiMapLayer.from_drivable_area_layer(static_map.raster_drivable_area_layer)
            static_map.raster_ground_height_layer = GroundHeightLayerRemote.from_file_remote(log_map_dirpath, log_map_dirpath_s3)

        return static_map

@dataclass(frozen=True)
class GroundHeightLayerRemote(GroundHeightLayer):
    """Rasterized ground height map layer.
    Stores the "ground_height_matrix" and also the array_Sim2_city: Sim(2) that produces takes point in city
    coordinates to numpy image/matrix coordinates, e.g. p_npyimage = array_Transformation_city * p_city
    """

    @classmethod
    def from_file_remote(cls, log_map_dirpath: Union[Path, UPath], log_map_dirpath_s3) -> GroundHeightLayer:
        """Load ground height values (w/ values at 30 cm resolution) from .npy file, and associated Sim(2) mapping.
        Note: ground height values are stored on disk as a float16 2d-array, but cast to float32 once loaded for
        compatibility with matplotlib.
        Args:
            log_map_dirpath: path to directory which contains map files associated with one specific log/scenario.
        Returns:
            The ground height map layer.
        Raises:
            RuntimeError: If raster ground height layer file is missing or Sim(2) mapping from city to image coordinates
                is missing.
        """
        # ground_height_npy_fpaths = sorted(log_map_dirpath.glob("*_ground_height_surface____*.npy"))
        ground_height_npy_fpaths = sorted(refile.smart_glob(log_map_dirpath_s3 + "/*_ground_height_surface____*.npy"))
        if not len(ground_height_npy_fpaths) == 1:
            raise RuntimeError("Raster ground height layer file is missing")

        # Sim2_json_fpaths = sorted(log_map_dirpath.glob("*___img_Sim2_city.json"))
        Sim2_json_fpaths = sorted(refile.smart_glob(log_map_dirpath_s3 + "/*___img_Sim2_city.json"))
        if not len(Sim2_json_fpaths) == 1:
            raise RuntimeError("Sim(2) mapping from city to image coordinates is missing")

        # load the file with rasterized values
        # with ground_height_npy_fpaths[0].open("rb") as f:
        with refile.smart_open(ground_height_npy_fpaths[0], "rb") as f:
            _bytes = f.read()
        ground_height_array: NDArrayFloat = np.load(BytesIO(_bytes))

        # array_Sim2_city = Sim2.from_json(Sim2_json_fpaths[0])
        with refile.smart_open(Sim2_json_fpaths[0], "r") as f:
            json_data = json.load(f)
        R: NDArrayFloat = np.array(json_data["R"]).reshape(2, 2)
        t: NDArrayFloat = np.array(json_data["t"]).reshape(2)
        s = float(json_data["s"])
        array_Sim2_city = Sim2(R, t, s)

        return cls(array=ground_height_array.astype(float), array_Sim2_city=array_Sim2_city)