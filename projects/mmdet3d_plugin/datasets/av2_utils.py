import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from av2.evaluation.detection.constants import (
    MAX_NORMALIZED_ASE,
    MAX_SCALE_ERROR,
    MAX_YAW_RAD_ERROR,
    MIN_AP,
    MIN_CDS,
    AffinityType,
    CompetitionCategories,
    DistanceType,
    FilterMetricType,
)
from av2.geometry.geometry import mat_to_xyz, quat_to_mat, wrap_angles
from av2.geometry.iou import iou_3d_axis_aligned
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt
import kornia.geometry.conversions as C
import torch
from torch import Tensor
from math import pi as PI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionCfg:

    # affinity_thresholds_m: Tuple[float, ...] = (4.0,)
    affinity_thresholds_m: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    affinity_type: AffinityType = AffinityType.CENTER
    categories: Tuple[str, ...] = tuple(x.value for x in CompetitionCategories)
    dataset_dir: Optional[Path] = None
    eval_only_roi_instances: bool = True
    filter_metric: FilterMetricType = FilterMetricType.EUCLIDEAN
    max_num_dts_per_category: int = 100
    eval_range_m: Tuple[float, ...] = (0.0, 150.0)
    num_recall_samples: int = 100
    tp_threshold_m: float = 2.0

    @property
    def metrics_defaults(self) -> Tuple[float, ...]:
        """Return the evaluation summary default values."""
        return (
            MIN_AP,
            self.tp_threshold_m,
            MAX_NORMALIZED_ASE,
            MAX_YAW_RAD_ERROR,
            MIN_CDS,
            MIN_AP,
        )

    @property
    def tp_normalization_terms(self) -> Tuple[float, ...]:
        """Return the normalization constants for ATE, ASE, and AOE."""
        return (
            self.tp_threshold_m,
            MAX_SCALE_ERROR,
            MAX_YAW_RAD_ERROR,
        )
        
def accumulate(
    dts: NDArrayFloat,
    gts: NDArrayFloat,
    cfg: DetectionCfg,
    avm: Optional[ArgoverseStaticMap] = None,
    city_SE3_ego: Optional[SE3] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    
    N, M = len(dts), len(gts)
    T, E = len(cfg.affinity_thresholds_m), 3

    # Sort the detections by score in _descending_ order.
    scores: NDArrayFloat = dts[..., -1]
    permutation: NDArrayInt = np.argsort(-scores).tolist()
    dts = dts[permutation]

    is_evaluated_dts: NDArrayBool = np.ones(N, dtype=bool)
    is_evaluated_gts: NDArrayBool = np.ones(M, dtype=bool)
    if avm is not None and city_SE3_ego is not None:
        is_evaluated_dts &= compute_objects_in_roi_mask(dts, city_SE3_ego, avm)
        is_evaluated_gts &= compute_objects_in_roi_mask(gts, city_SE3_ego, avm)

    is_evaluated_dts &= compute_evaluated_dts_mask(dts[..., :3], cfg)
    is_evaluated_gts &= compute_evaluated_gts_mask(gts[..., :3], gts[..., -1], cfg)

    # Initialize results array.
    dts_augmented: NDArrayFloat = np.zeros((N, T + E + 1))
    gts_augmented: NDArrayFloat = np.zeros((M, T + E + 1))

    # `is_evaluated` boolean flag is always the last column of the array.
    dts_augmented[is_evaluated_dts, -1] = True
    gts_augmented[is_evaluated_gts, -1] = True

    if is_evaluated_dts.sum() > 0 and is_evaluated_gts.sum() > 0:
        # Compute true positives by assigning detections and ground truths.
        dts_assignments, gts_assignments = assign(dts[is_evaluated_dts], gts[is_evaluated_gts], cfg)
        dts_augmented[is_evaluated_dts, :-1] = dts_assignments
        gts_augmented[is_evaluated_gts, :-1] = gts_assignments

    # Permute the detections according to the original ordering.
    outputs: Tuple[NDArrayInt, NDArrayInt] = np.unique(permutation, return_index=True)  # type: ignore
    _, inverse_permutation = outputs
    dts_augmented = dts_augmented[inverse_permutation]
    return dts_augmented, gts_augmented

def assign(dts: NDArrayFloat, gts: NDArrayFloat, cfg: DetectionCfg) -> Tuple[NDArrayFloat, NDArrayFloat]:

    affinity_matrix = compute_affinity_matrix(dts[..., :3], gts[..., :3], cfg.affinity_type)

    # Get the GT label for each max-affinity GT label, detection pair.
    idx_gts = affinity_matrix.argmax(axis=1)[None]

    # The affinity matrix is an N by M matrix of the detections and ground truth labels respectively.
    # We want to take the corresponding affinity for each of the initial assignments using `gt_matches`.
    # The following line grabs the max affinity for each detection to a ground truth label.
    affinities: NDArrayFloat = np.take_along_axis(affinity_matrix.transpose(), idx_gts, axis=0)[0]  # type: ignore

    # Find the indices of the _first_ detection assigned to each GT.
    assignments: Tuple[NDArrayInt, NDArrayInt] = np.unique(idx_gts, return_index=True)  # type: ignore
    idx_gts, idx_dts = assignments

    T, E = len(cfg.affinity_thresholds_m), 3
    dts_metrics: NDArrayFloat = np.zeros((len(dts), T + E))
    dts_metrics[:, T:] = cfg.metrics_defaults[1:4]
    gts_metrics: NDArrayFloat = np.zeros((len(gts), T + E))
    gts_metrics[:, T:] = cfg.metrics_defaults[1:4]
    for i, threshold_m in enumerate(cfg.affinity_thresholds_m):
        is_tp: NDArrayBool = affinities[idx_dts] > -threshold_m

        dts_metrics[idx_dts[is_tp], i] = True
        gts_metrics[idx_gts, i] = True

        if threshold_m != cfg.tp_threshold_m:
            continue  # Skip if threshold isn't the true positive threshold.
        if not np.any(is_tp):
            continue  # Skip if no true positives exist.

        idx_tps_dts: NDArrayInt = idx_dts[is_tp]
        idx_tps_gts: NDArrayInt = idx_gts[is_tp]

        tps_dts = dts[idx_tps_dts]
        tps_gts = gts[idx_tps_gts]

        translation_errors = distance(tps_dts[:, :3], tps_gts[:, :3], DistanceType.TRANSLATION)
        scale_errors = distance(tps_dts[:, 3:6], tps_gts[:, 3:6], DistanceType.SCALE)
        orientation_errors = distance(tps_dts[:, 6:10], tps_gts[:, 6:10], DistanceType.ORIENTATION)
        dts_metrics[idx_tps_dts, T:] = np.stack((translation_errors, scale_errors, orientation_errors), axis=-1)
    return dts_metrics, gts_metrics

def distance(dts: NDArrayFloat, gts: NDArrayFloat, metric: DistanceType) -> NDArrayFloat:

    if metric == DistanceType.TRANSLATION:
        translation_errors: NDArrayFloat = np.linalg.norm(dts - gts, axis=1)  # type: ignore
        return translation_errors
    elif metric == DistanceType.SCALE:
        scale_errors: NDArrayFloat = 1 - iou_3d_axis_aligned(dts, gts)
        return scale_errors
    elif metric == DistanceType.ORIENTATION:
        yaws_dts: NDArrayFloat = mat_to_xyz(quat_to_mat(dts))[..., 2]
        yaws_gts: NDArrayFloat = mat_to_xyz(quat_to_mat(gts))[..., 2]
        orientation_errors = wrap_angles(yaws_dts - yaws_gts)
        return orientation_errors
    else:
        raise NotImplementedError("This distance metric is not implemented!")

def compute_affinity_matrix(dts: NDArrayFloat, gts: NDArrayFloat, metric: AffinityType) -> NDArrayFloat:

    if metric == AffinityType.CENTER:
        dts_xy_m = dts
        gts_xy_m = gts
        affinities: NDArrayFloat = -cdist(dts_xy_m, gts_xy_m)
    else:
        raise NotImplementedError("This affinity metric is not implemented!")
    return affinities

def compute_evaluated_dts_mask(
    xyz_m_ego: NDArrayFloat,
    cfg: DetectionCfg,
) -> NDArrayBool:

    is_evaluated: NDArrayBool
    if len(xyz_m_ego) == 0:
        is_evaluated = np.zeros((0,), dtype=bool)
        return is_evaluated
    norm: NDArrayFloat = np.linalg.norm(xyz_m_ego, axis=1)  # type: ignore
    # is_evaluated = norm < cfg.max_range_m
    is_evaluated = np.logical_and(norm > cfg.eval_range_m[0], norm < cfg.eval_range_m[1])

    cumsum: NDArrayInt = np.cumsum(is_evaluated)
    max_idx_arr: NDArrayInt = np.where(cumsum > cfg.max_num_dts_per_category)[0]
    if len(max_idx_arr) > 0:
        max_idx = max_idx_arr[0]
        is_evaluated[max_idx:] = False  # type: ignore
    return is_evaluated

def compute_evaluated_gts_mask(
    xyz_m_ego: NDArrayFloat,
    num_interior_pts: NDArrayInt,
    cfg: DetectionCfg,
) -> NDArrayBool:

    is_evaluated: NDArrayBool
    if len(xyz_m_ego) == 0:
        is_evaluated = np.zeros((0,), dtype=bool)
        return is_evaluated
    norm: NDArrayFloat = np.linalg.norm(xyz_m_ego, axis=1)  # type: ignore
    is_evaluated_range = np.logical_and(norm > cfg.eval_range_m[0], norm < cfg.eval_range_m[1])
    is_evaluated = np.logical_and(is_evaluated_range, num_interior_pts > 0)
    # is_evaluated = np.logical_and(norm < cfg.eval_range_m[1], num_interior_pts > 0)
    
    return is_evaluated

def compute_objects_in_roi_mask(cuboids_ego: NDArrayFloat, city_SE3_ego: SE3, avm: ArgoverseStaticMap) -> NDArrayBool:

    is_within_roi: NDArrayBool
    if len(cuboids_ego) == 0:
        is_within_roi = np.zeros((0,), dtype=bool)
        return is_within_roi
    cuboid_list_ego: CuboidList = CuboidList([Cuboid.from_numpy(params) for params in cuboids_ego])
    cuboid_list_city = cuboid_list_ego.transform(city_SE3_ego)
    cuboid_list_vertices_m_city = cuboid_list_city.vertices_m

    is_within_roi = avm.get_raster_layer_points_boolean(
        cuboid_list_vertices_m_city.reshape(-1, 3)[..., :2], RasterLayerType.ROI
    )
    is_within_roi = is_within_roi.reshape(-1, 8)
    is_within_roi = is_within_roi.any(axis=1)
    return is_within_roi


@torch.jit.script
def xyz_to_quat(xyz_rad: Tensor) -> Tensor:
    """Convert euler angles (xyz - pitch, roll, yaw) to scalar first quaternions.

    Args:
        xyz_rad: (...,3) Tensor of roll, pitch, and yaw in radians.

    Returns:
        (...,4) Scalar first quaternions (wxyz).
    """
    x_rad = xyz_rad[..., 0]
    y_rad = xyz_rad[..., 1]
    z_rad = xyz_rad[..., 2]

    cy = torch.cos(z_rad * 0.5)
    sy = torch.sin(z_rad * 0.5)
    cp = torch.cos(y_rad * 0.5)
    sp = torch.sin(y_rad * 0.5)
    cr = torch.cos(x_rad * 0.5)
    sr = torch.sin(x_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat_wxyz = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat_wxyz


@torch.jit.script
def yaw_to_quat(yaw_rad: Tensor) -> Tensor:
    """Convert yaw (rotation about the vertical axis) to scalar first quaternions.

    Args:
        yaw_rad: (...,1) Rotations about the z-axis.

    Returns:
        (...,4) scalar first quaternions (wxyz).
    """
    xyz_rad = torch.zeros_like(yaw_rad)[..., None].repeat_interleave(3, dim=-1)
    xyz_rad[..., -1] = yaw_rad
    quat_wxyz: Tensor = xyz_to_quat(xyz_rad)
    return quat_wxyz

