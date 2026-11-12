"""Incremental registration, feature tracks, and multi-view triangulation."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from initialize import (
    build_pose_matrix,
    choose_initial_pair,
    estimate_relative_pose,
    project_points,
    triangulate_points,
    triangulation_mask,
)
from match import oriented_matches


def solve_pnp_pose(points_3d: np.ndarray, points_2d: np.ndarray, K: np.ndarray) -> tuple:
    if len(points_3d) < 12:
        raise ValueError("PnP requires at least twelve correspondences")
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.ascontiguousarray(points_3d, dtype=np.float64),
        np.ascontiguousarray(points_2d, dtype=np.float64), K, None,
        iterationsCount=300, reprojectionError=3.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 12:
        raise ValueError("PnP RANSAC found insufficient inliers")
    selected = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(points_3d[selected], points_2d[selected], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    pose = build_pose_matrix(R, tvec)
    errors = np.linalg.norm(project_points(points_3d, K, pose) - points_2d, axis=1)
    depth = (points_3d @ R.T + tvec.ravel())[:, 2]
    selected = np.flatnonzero((errors <= 3.0) & (depth > 0))
    if len(selected) < 12:
        raise ValueError("Refined PnP pose failed geometric checks")
    return pose, selected


@dataclass
class Reconstruction:
    keypoints: list
    pair_matches: dict
    K: np.ndarray
    poses: dict = field(default_factory=dict)
    points: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    feature_to_point: list = field(default_factory=list)
    registration_log: list = field(default_factory=list)

    def initialize(self) -> tuple:
        pair, pose, points, matches = choose_initial_pair(self.keypoints, self.pair_matches, self.K)
        self.feature_to_point = [{} for _ in self.keypoints]
        self.poses = {0: np.eye(4, dtype=np.float64), pair[1]: pose}
        for point, match in zip(points, matches):
            self.add_point(point, 0, match.queryIdx, pair[1], match.trainIdx)
        return pair, points.copy()

    def add_point(self, point: np.ndarray, i: int, ki: int, j: int, kj: int) -> None:
        point_id = len(self.points)
        self.points.append(point)
        self.tracks.append([(i, ki), (j, kj)])
        self.feature_to_point[i][ki] = point_id
        self.feature_to_point[j][kj] = point_id

    def correspondences(self, image_idx: int) -> tuple:
        candidates = {}
        for ref in sorted(self.poses):
            for ref_kp, new_kp in oriented_matches(self.pair_matches, ref, image_idx):
                pid = self.feature_to_point[ref].get(ref_kp)
                if pid is not None:
                    votes = candidates.setdefault(new_kp, {})
                    votes[pid] = votes.get(pid, 0) + 1
        # A feature may name one point, and a point may appear once per image.
        unique = [(kp, next(iter(votes)), next(iter(votes.values())))
                  for kp, votes in candidates.items() if len(votes) == 1]
        unique.sort(key=lambda item: (-item[2], item[0]))
        used, pairs = set(), []
        for kp, pid, _ in unique:
            if pid not in used:
                used.add(pid)
                pairs.append((kp, pid))
        if not pairs:
            return np.empty((0, 3)), np.empty((0, 2)), []
        xyz = np.float64([self.points[pid] for kp, pid in pairs])
        xy = np.float64([self.keypoints[image_idx][kp].pt for kp, pid in pairs])
        return xyz, xy, pairs

    def register_next(self, pose_method: str = "pnp") -> int | None:
        pending = [i for i in range(len(self.keypoints)) if i not in self.poses]
        candidates = [(i, self.correspondences(i)) for i in pending]
        candidates.sort(key=lambda item: len(item[1][0]), reverse=True)
        for index, (xyz, xy, pairs) in candidates:
            try:
                if pose_method == "pnp":
                    pose, inliers = solve_pnp_pose(xyz, xy, self.K)
                else:
                    pose = self.essential_pose(index)
                    errors = np.linalg.norm(project_points(xyz, self.K, pose) - xy, axis=1)
                    depth = (xyz @ pose[:3, :3].T + pose[:3, 3])[:, 2]
                    inliers = np.flatnonzero((errors <= 3.0) & (depth > 0))
            except (ValueError, cv2.error):
                continue
            self.poses[index] = pose
            for selected in inliers:
                kp, pid = pairs[selected]
                self.feature_to_point[index][kp] = pid
                self.tracks[pid].append((index, kp))
            self.triangulate_new_points(index)
            self.registration_log.append({"image_index": index, "correspondences": len(xyz),
                                          "inliers": len(inliers), "points": len(self.points)})
            return index
        return None

    def essential_pose(self, image_idx: int) -> np.ndarray:
        """Ablation using unit-baseline relative poses without PnP scale recovery."""
        refs = sorted(self.poses, key=lambda i: len(oriented_matches(self.pair_matches, i, image_idx)), reverse=True)
        for ref in refs:
            pairs = oriented_matches(self.pair_matches, ref, image_idx)
            if len(pairs) < 30:
                continue
            xy1 = np.float64([self.keypoints[ref][a].pt for a, b in pairs])
            xy2 = np.float64([self.keypoints[image_idx][b].pt for a, b in pairs])
            try:
                R, t, inliers = estimate_relative_pose(xy1, xy2, self.K)
            except (ValueError, cv2.error):
                continue
            if inliers.sum() >= 20:
                return build_pose_matrix(R, t) @ self.poses[ref]
        raise ValueError("No reliable essential-matrix connection")

    def triangulate_new_points(self, image_idx: int) -> None:
        for ref in sorted(self.poses):
            if ref == image_idx:
                continue
            pairs = [(a, b) for a, b in oriented_matches(self.pair_matches, ref, image_idx)
                     if a not in self.feature_to_point[ref] and b not in self.feature_to_point[image_idx]]
            if not pairs:
                continue
            xy1 = np.float64([self.keypoints[ref][a].pt for a, b in pairs])
            xy2 = np.float64([self.keypoints[image_idx][b].pt for a, b in pairs])
            points = triangulate_points(xy1, xy2, self.K, self.poses[ref], self.poses[image_idx])
            valid = triangulation_mask(points, xy1, xy2, self.K, self.poses[ref], self.poses[image_idx])
            for point, (a, b) in zip(points[valid], [pair for pair, ok in zip(pairs, valid) if ok]):
                self.add_point(point, ref, a, image_idx, b)

    def observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = [(image, pid, self.keypoints[image][kp].pt)
                for pid, track in enumerate(self.tracks) for image, kp in track]
        cameras = np.int32([r[0] for r in rows])
        points = np.int32([r[1] for r in rows])
        xy = np.float64([r[2] for r in rows])
        return cameras, points, xy
