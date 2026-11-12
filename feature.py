"""Image loading and SIFT extraction with consistently scaled intrinsics."""

from pathlib import Path

import cv2
import numpy as np

WORLD_FRAME_IMAGE_NAME = "DJI_20200223_163225_243.jpg"


def load_image_paths(image_dir: str | Path, stride: int = 1) -> list[Path]:
    if stride < 1:
        raise ValueError("Image stride must be positive")
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) < 2:
        raise ValueError(f"At least two images are required in {image_dir}")
    anchor = next((i for i, p in enumerate(paths) if p.name == WORLD_FRAME_IMAGE_NAME), 0)
    selected = (paths[anchor:] + paths[:anchor])[::stride]
    if len(selected) < 2:
        raise ValueError("Image stride leaves fewer than two images")
    return selected


def load_images(image_paths: list[Path], K: np.ndarray, max_size: int) -> tuple[list[np.ndarray], np.ndarray]:
    images = []
    original_shape = None
    scaled_K = K.astype(np.float64).copy()
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        h, w = image.shape[:2]
        if original_shape is None:
            original_shape = (h, w)
            scale = min(1.0, max_size / max(h, w))
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            scaled_K[0] *= size[0] / w
            scaled_K[1] *= size[1] / h
        if (h, w) != original_shape:
            raise ValueError("Images must share the calibrated image dimensions")
        images.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
    return images, scaled_K


def extract_sift_features(images: list[np.ndarray], max_features: int = 6000) -> tuple[list, list[np.ndarray]]:
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = [], []
    for i, image in enumerate(images):
        kp, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        keypoints.append(list(kp))
        descriptors.append(desc if desc is not None else np.empty((0, 128), np.float32))
        if (i + 1) % 10 == 0 or i + 1 == len(images):
            print(f"SIFT {i + 1}/{len(images)}", flush=True)
    return keypoints, descriptors


def save_keypoint_visualization(image: np.ndarray, keypoints: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if not cv2.imwrite(str(path), vis):
        raise OSError(f"Cannot write {path}")
