"""Calibrated two-view initialization and triangulation quality checks."""

import cv2
import numpy as np


def build_pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = np.asarray(t).reshape(3)
    return T_cw


def estimate_relative_pose(xy1: np.ndarray, xy2: np.ndarray, K: np.ndarray) -> tuple:
    if len(xy1) < 8:
        raise ValueError("Essential-matrix estimation needs at least eight matches")
    E, mask = cv2.findEssentialMat(xy1, xy2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None or E.shape != (3, 3):
        raise ValueError("Essential-matrix estimation failed")
    _, R, t, pose_mask = cv2.recoverPose(E, xy1, xy2, K, mask=mask.copy())
    return R, t.ravel(), pose_mask.ravel() > 0


def project_points(points_3d: np.ndarray, K: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    camera = points_3d @ T_cw[:3, :3].T + T_cw[:3, 3]
    homogeneous = camera @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return homogeneous[:, :2] / homogeneous[:, 2:3]


def triangulate_points(xy1: np.ndarray, xy2: np.ndarray, K: np.ndarray,
                       T_cw_1: np.ndarray, T_cw_2: np.ndarray) -> np.ndarray:
    homogeneous = cv2.triangulatePoints(K @ T_cw_1[:3], K @ T_cw_2[:3], xy1.T, xy2.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (homogeneous[:3] / homogeneous[3]).T


def triangulation_mask(points_3d: np.ndarray, xy1: np.ndarray, xy2: np.ndarray,
                       K: np.ndarray, T_cw_1: np.ndarray, T_cw_2: np.ndarray,
                       max_error: float = 2.0, min_angle: float = 1.5) -> np.ndarray:
    depth1 = (points_3d @ T_cw_1[:3, :3].T + T_cw_1[:3, 3])[:, 2]
    depth2 = (points_3d @ T_cw_2[:3, :3].T + T_cw_2[:3, 3])[:, 2]
    c1 = -T_cw_1[:3, :3].T @ T_cw_1[:3, 3]
    c2 = -T_cw_2[:3, :3].T @ T_cw_2[:3, 3]
    rays1, rays2 = points_3d - c1, points_3d - c2
    denominator = np.maximum(np.linalg.norm(rays1, axis=1) * np.linalg.norm(rays2, axis=1), 1e-15)
    cosine = np.clip(np.sum(rays1 * rays2, axis=1) / denominator, -1, 1)
    angle = np.degrees(np.arccos(cosine))
    error1 = np.linalg.norm(project_points(points_3d, K, T_cw_1) - xy1, axis=1)
    error2 = np.linalg.norm(project_points(points_3d, K, T_cw_2) - xy2, axis=1)
    return (np.isfinite(points_3d).all(axis=1) & (depth1 > 0) & (depth2 > 0)
            & (angle >= min_angle) & (error1 <= max_error) & (error2 <= max_error))


def choose_initial_pair(keypoints: list, pair_matches: dict, K: np.ndarray) -> tuple:
    """Choose an anchor pair by valid triangulations and parallax."""
    candidates = sorted((p for p in pair_matches if p[0] == 0),
                        key=lambda p: len(pair_matches[p]), reverse=True)
    best, best_score = None, 0
    for pair in candidates:
        matches = pair_matches[pair]
        if len(matches) < 30:
            continue
        xy1 = np.float64([keypoints[pair[0]][m.queryIdx].pt for m in matches])
        xy2 = np.float64([keypoints[pair[1]][m.trainIdx].pt for m in matches])
        try:
            R, t, inliers = estimate_relative_pose(xy1, xy2, K)
        except (ValueError, cv2.error):
            continue
        pose = build_pose_matrix(R, t)
        points = triangulate_points(xy1, xy2, K, np.eye(4), pose)
        valid = inliers & triangulation_mask(points, xy1, xy2, K, np.eye(4), pose)
        score = int(valid.sum())
        if score > best_score:
            best, best_score = (pair, pose, points[valid], [m for m, ok in zip(matches, valid) if ok]), score
    if best_score < 20:
        raise ValueError("No anchor pair has enough finite, positive-depth points with parallax")
    return best
