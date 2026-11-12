"""Run the six-module dinosaur Structure-from-Motion pipeline."""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import scipy

from ba import run_bundle_adjustment
from feature import extract_sift_features, load_image_paths, load_images, save_keypoint_visualization
from initialize import project_points, triangulate_points, triangulation_mask
from match import build_pair_matches, save_match_visualization
from reconstruct import Reconstruction

ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_point_cloud_ply(points_3d: np.ndarray, colors: np.ndarray, path: Path) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points_3d, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f"Cannot write {path}")


def reprojection_stats(poses: np.ndarray, points: np.ndarray, cameras: np.ndarray,
                       point_ids: np.ndarray, observed: np.ndarray, K: np.ndarray) -> dict:
    camera_xyz = np.einsum("nij,nj->ni", poses[cameras, :3, :3], points[point_ids]) + poses[cameras, :3, 3]
    pixels = camera_xyz @ K.T
    errors = np.linalg.norm(pixels[:, :2] / pixels[:, 2:3] - observed, axis=1)
    return {"mean_px": float(np.mean(errors)), "median_px": float(np.median(errors)),
            "p90_px": float(np.percentile(errors, 90)), "observations": len(errors),
            "positive_depth_fraction": float(np.mean(camera_xyz[:, 2] > 0))}


def sample_point_colors(tracks: list, keypoints: list, images: list) -> np.ndarray:
    colors = []
    for track in tracks:
        samples = []
        for image, kp in track:
            x, y = np.rint(keypoints[image][kp].pt).astype(int)
            h, w = images[image].shape[:2]
            samples.append(images[image][np.clip(y, 0, h - 1), np.clip(x, 0, w - 1), ::-1] / 255.0)
        colors.append(np.median(samples, axis=0))
    return np.asarray(colors, dtype=np.float64)


def export_demo(demo_dir: Path, points: np.ndarray, colors: np.ndarray, initial: np.ndarray,
                poses: np.ndarray, images: list, image_paths: list, metrics: dict, initialization: dict,
                keypoints: list, pair_matches: dict, pair: tuple, output: Path) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(points), min(len(points), 50000), replace=False))
    rotations = poses[:, :3, :3]
    centers = -np.einsum("nji,nj->ni", rotations, poses[:, :3, 3])
    directions = rotations[:, 2, :]
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    target = np.linalg.lstsq(projectors.sum(axis=0),
                             np.einsum("nij,nj->i", projectors, centers), rcond=None)[0]
    scene = {
        "view_target": target.tolist(), "view_position": centers[0].tolist(),
        "points": np.round(points[sample], 5).tolist(),
        "colors": np.rint(colors[sample] * 255).astype(int).tolist(),
        "initial_points": np.round(initial, 5).tolist(),
        "initialization": initialization,
        "poses": np.round(poses, 8).tolist(),
        "image_names": [p.name for p in image_paths],
        "metrics": metrics,
    }
    write_json(demo_dir / "scene.json", scene)
    save_keypoint_visualization(images[0], keypoints[0], output / "keypoints.jpg")
    save_match_visualization(images, keypoints, pair_matches, pair, output / "matches.jpg")
    for name in ["keypoints.jpg", "matches.jpg"]:
        image = cv2.imread(str(output / name))
        h, w = image.shape[:2]
        cv2.imwrite(str(demo_dir / name), cv2.resize(image, (1200, round(h * 1200 / w))))
    thumb_dir = demo_dir / "frames"
    thumb_dir.mkdir(exist_ok=True)
    for index, image in enumerate(images):
        h, w = image.shape[:2]
        cv2.imwrite(str(thumb_dir / f"{index:03d}.jpg"), cv2.resize(image, (480, round(h * 480 / w))))
    # Discard thumbnails from a previous, larger dataset export.
    for old in thumb_dir.glob("*.jpg"):
        if old.stem.isdigit() and int(old.stem) >= len(images):
            old.unlink()
    save_point_cloud_ply(points, colors, demo_dir / "reconstruction.ply")
    np.savetxt(demo_dir / "camera_poses.txt", poses.reshape(len(poses), 16), fmt="%.10f")
    write_json(demo_dir / "metrics.json", metrics)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    cv2.setRNGSeed(args.seed)
    cv2.setNumThreads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Remove completion markers before starting a new run in the same directory.
    for old in [output / "metrics.json", output / "final_points.ply", output / "final_poses.txt",
                output / "524370910025.ply", output / "524370910025.txt"]:
        old.unlink(missing_ok=True)
    image_paths = load_image_paths(args.data / "images", args.stride)
    original_K = np.loadtxt(args.data / "camera_intrinsic.txt", dtype=np.float64).reshape(3, 3)
    if not np.isfinite(original_K).all() or min(original_K[0, 0], original_K[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    images, K = load_images(image_paths, original_K, args.max_size)
    print(f"Loaded {len(images)} images at {images[0].shape[1]} x {images[0].shape[0]}", flush=True)
    feature_start = time.perf_counter()
    keypoints, descriptors = extract_sift_features(images, args.features)
    feature_seconds = time.perf_counter() - feature_start
    match_start = time.perf_counter()
    pair_matches = build_pair_matches(descriptors, keypoints, args.match_gap)
    match_seconds = time.perf_counter() - match_start
    reconstruction = Reconstruction(keypoints, pair_matches, K)
    pair, initial = reconstruction.initialize()
    initialization = {"pair": list(pair), "poses": [reconstruction.poses[i].tolist() for i in pair],
                      "point_count": len(initial)}
    write_json(output / "initialization.json", initialization)
    print(f"Initialized pair {pair} with {len(initial)} points", flush=True)
    save_point_cloud_ply(initial, np.full(initial.shape, 0.65), output / "initial_points.ply")
    save_match_visualization(images, keypoints, pair_matches, pair, output / "matches.jpg")
    save_keypoint_visualization(images[0], keypoints[0], output / "keypoints.jpg")
    while len(reconstruction.poses) < len(images):
        index = reconstruction.register_next(args.pose_method)
        if index is None:
            missing = [image_paths[i].name for i in range(len(images)) if i not in reconstruction.poses]
            write_json(output / "failure.json", {"registered": len(reconstruction.poses), "missing_images": missing})
            raise RuntimeError(f"Registered {len(reconstruction.poses)}/{len(images)} images. See {output / 'failure.json'}")
        print(f"Registered {len(reconstruction.poses)}/{len(images)} images, {len(reconstruction.points)} points", flush=True)
    (output / "failure.json").unlink(missing_ok=True)
    poses = np.array([reconstruction.poses[i] for i in range(len(images))])
    points = np.asarray(reconstruction.points, dtype=np.float64)
    camera_indices, point_indices, observations = reconstruction.observations()
    colors = sample_point_colors(reconstruction.tracks, keypoints, images)
    np.savez_compressed(output / "pre_ba.npz", poses=poses, points=points, colors=colors,
                        camera_indices=camera_indices, point_indices=point_indices, observations=observations, K=K)
    before = reprojection_stats(poses, points, camera_indices, point_indices, observations, K)
    print(f"Before BA: mean {before['mean_px']:.4f} px, median {before['median_px']:.4f} px", flush=True)
    if args.no_ba:
        ba_info = {"enabled": False, "converged": None, "seconds": 0}
    else:
        poses, points, ba_info = run_bundle_adjustment(poses, points, camera_indices, point_indices,
                                                      observations, K, args.ba_evaluations)
    after = reprojection_stats(poses, points, camera_indices, point_indices, observations, K)
    if not np.isfinite(points).all() or not np.isfinite(poses).all():
        raise RuntimeError("Non-finite reconstruction")
    if not np.allclose(poses[0], np.eye(4), atol=1e-10):
        raise RuntimeError("The reference camera moved")
    if after["positive_depth_fraction"] < 0.99:
        raise RuntimeError("Too many observations lie behind their cameras")
    # Check every verified image-pair match, independent of the BA track set.
    pairwise_total, pairwise_valid = 0, 0
    for (i, j), matches in pair_matches.items():
        if not matches:
            continue
        xy1 = np.float64([keypoints[i][m.queryIdx].pt for m in matches])
        xy2 = np.float64([keypoints[j][m.trainIdx].pt for m in matches])
        triangulated = triangulate_points(xy1, xy2, K, poses[i], poses[j])
        valid = triangulation_mask(triangulated, xy1, xy2, K, poses[i], poses[j], min_angle=0)
        pairwise_total += len(matches)
        pairwise_valid += int(valid.sum())
    # Reject points whose optimized observations exceed the geometric threshold.
    max_errors = np.zeros(len(points))
    for i in range(len(poses)):
        selected = camera_indices == i
        ids = point_indices[selected]
        errors = np.linalg.norm(project_points(points[ids], K, poses[i]) - observations[selected], axis=1)
        np.maximum.at(max_errors, ids, errors)
    keep = max_errors <= 4.0
    final_points, final_colors = points[keep], colors[keep]
    if not len(final_points):
        raise RuntimeError("No points survived the final reprojection check")
    save_point_cloud_ply(final_points, final_colors, output / "final_points.ply")
    np.savetxt(output / "final_poses.txt", poses.reshape(len(poses), 16), fmt="%.10f")
    save_point_cloud_ply(final_points, final_colors, output / f"{args.student_id}.ply")
    np.savetxt(output / f"{args.student_id}.txt", poses.reshape(len(poses), 16), fmt="%.10f")
    np.savez_compressed(output / "reconstruction.npz", poses=poses, points=points, colors=colors,
                        camera_indices=camera_indices, point_indices=point_indices, observations=observations,
                        K=K, export_mask=keep)
    metrics = {
        "images": len(images), "registered_images": len(poses), "initial_pair": list(pair),
        "initial_points": len(initial), "points_before_filter": len(points), "exported_points": len(final_points),
        "mean_track_length": float(len(observations) / len(points)),
        "pairwise_consistency": {"tested_matches": pairwise_total, "consistent_matches": pairwise_valid,
                                 "fraction": pairwise_valid / pairwise_total,
                                 "threshold_px": 2.0, "positive_depth_required": True},
        "features": sum(map(len, keypoints)), "matched_pairs": len(pair_matches),
        "verified_matches": sum(map(len, pair_matches.values())),
        "processing_size": [images[0].shape[1], images[0].shape[0]], "intrinsics": K.tolist(),
        "image_order": [p.name for p in image_paths], "before_ba": before, "after_ba": after,
        "bundle_adjustment": ba_info, "configuration": {
            "max_size": args.max_size, "max_features": args.features, "match_gap": args.match_gap,
            "stride": args.stride, "pose_method": args.pose_method, "seed": args.seed,
            "ba_evaluations": args.ba_evaluations, "no_ba": args.no_ba},
        "timing_seconds": {"features": round(feature_seconds, 3), "matching": round(match_seconds, 3),
                           "pipeline": round(time.perf_counter() - started, 3)},
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "open3d": o3d.__version__},
        "registration": reconstruction.registration_log,
    }
    write_json(output / "metrics.json", metrics)
    if args.demo_dir is not None:
        export_demo(args.demo_dir, final_points, final_colors, initial, poses, images, image_paths,
                    metrics, initialization, keypoints, pair_matches, pair, output)
    print(f"Finished: {len(poses)} cameras, {len(final_points)} points, mean error {after['mean_px']:.4f} px", flush=True)
    print(f"Results: {output}", flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", default="reconstruction", help="Basename for assignment-format PLY/TXT exports")
    parser.add_argument("--data", type=Path, default=ROOT / "重建所需数据")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--demo-dir", type=Path, help="Export viewer assets into this directory")
    parser.add_argument("--max-size", type=int, default=1600)
    parser.add_argument("--features", type=int, default=6000)
    parser.add_argument("--match-gap", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ba-evaluations", type=int, default=100)
    parser.add_argument("--no-ba", action="store_true")
    parser.add_argument("--pose-method", choices=["pnp", "essential"], default="pnp")
    args = parser.parse_args()
    if min(args.max_size, args.features, args.match_gap, args.stride, args.ba_evaluations) < 1:
        parser.error("Size, feature count, gap, stride and evaluation budget must be positive")
    if not args.student_id or Path(args.student_id).name != args.student_id or args.student_id in {".", ".."}:
        parser.error("Student ID must be a filename without directory components")
    return args


if __name__ == "__main__":
    run(parse_args())
