"""Sparse robust bundle adjustment with analytic projection derivatives."""

import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from initialize import build_pose_matrix


def run_bundle_adjustment(camera_poses: np.ndarray, points_3d: np.ndarray,
                          camera_indices: np.ndarray, point_indices: np.ndarray,
                          observations_2d: np.ndarray, K: np.ndarray,
                          max_nfev: int = 100) -> tuple:
    """Fix camera zero and one translation component to remove similarity gauge."""
    start = time.perf_counter()
    camera_params = np.array([np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]] for T in camera_poses])
    num_cameras = len(camera_params)
    point_offset = num_cameras * 6
    initial = np.r_[camera_params.ravel(), points_3d.ravel()]
    free = np.ones(len(initial), dtype=bool)
    free[:6] = False
    scale_camera = 1 + np.argmax(np.linalg.norm(camera_params[1:, 3:], axis=1))
    scale_component = int(np.argmax(np.abs(camera_params[scale_camera, 3:])))
    scale_parameter = int(scale_camera * 6 + 3 + scale_component)
    if abs(initial[scale_parameter]) < 1e-10:
        raise ValueError("Bundle adjustment requires a nonzero baseline")
    free[scale_parameter] = False
    groups = [np.flatnonzero(camera_indices == i) for i in range(num_cameras)]
    rows = (2 * np.arange(len(observations_2d))[:, None] + np.arange(2)).ravel()
    camera_cols = (6 * camera_indices[:, None] + np.arange(6))
    camera_cols = np.repeat(camera_cols, 2, axis=0)
    point_cols = point_offset + 3 * point_indices[:, None] + np.arange(3)
    point_cols = np.repeat(point_cols, 2, axis=0)
    jac_rows = np.r_[np.repeat(rows, 6), np.repeat(rows, 3)]
    jac_cols = np.r_[camera_cols.ravel(), point_cols.ravel()]

    def unpack(parameters: np.ndarray) -> tuple:
        full = initial.copy()
        full[free] = parameters
        return full[:point_offset].reshape(-1, 6), full[point_offset:].reshape(-1, 3)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        cameras, points = unpack(parameters)
        rotations = np.array([cv2.Rodrigues(c[:3])[0] for c in cameras])
        camera_xyz = np.einsum("nij,nj->ni", rotations[camera_indices], points[point_indices])
        camera_xyz += cameras[camera_indices, 3:]
        pixels = camera_xyz @ K.T
        projected = pixels[:, :2] / pixels[:, 2:3]
        return (projected - observations_2d).ravel()

    def jacobian(parameters: np.ndarray):
        cameras, points = unpack(parameters)
        camera_derivatives = np.empty((len(observations_2d), 2, 6))
        point_derivatives = np.empty((len(observations_2d), 2, 3))
        for index, selected in enumerate(groups):
            if not len(selected):
                continue
            camera = cameras[index]
            _, derivative = cv2.projectPoints(points[point_indices[selected]], camera[:3], camera[3:], K, None)
            derivative = derivative.reshape(len(selected), 2, -1)
            camera_derivatives[selected] = derivative[:, :, :6]
            rotation = cv2.Rodrigues(camera[:3])[0]
            point_derivatives[selected] = derivative[:, :, 3:6] @ rotation
        values = np.r_[camera_derivatives.ravel(), point_derivatives.ravel()]
        sparse = coo_matrix((values, (jac_rows, jac_cols)), shape=(len(rows), len(initial))).tocsr()
        return sparse[:, free]

    iterations = 0

    def progress(parameters: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1
        if iterations % 5 == 0:
            print(f"BA iteration {iterations}, elapsed {time.perf_counter() - start:.0f}s", flush=True)

    result = least_squares(
        residuals, initial[free], jac=jacobian, method="trf", loss="soft_l1", f_scale=1.0,
        x_scale="jac", max_nfev=max_nfev, ftol=1e-4, xtol=1e-7, gtol=1e-6,
        tr_solver="lsmr", tr_options={"atol": 1e-4, "btol": 1e-4, "maxiter": 100},
        callback=progress,
    )
    cameras, points = unpack(result.x)
    poses = np.array([build_pose_matrix(cv2.Rodrigues(c[:3])[0], c[3:]) for c in cameras])
    information = {
        "enabled": True, "converged": bool(result.success), "status": int(result.status),
        "message": result.message, "evaluations": int(result.nfev), "iterations": iterations,
        "seconds": round(time.perf_counter() - start, 3),
        "scale_camera_index": int(scale_camera), "scale_translation_component": scale_component,
        "cost": float(result.cost), "optimality": float(result.optimality),
    }
    print(f"BA finished after {result.nfev} evaluations: {result.message}", flush=True)
    return poses, points, information
