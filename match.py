"""Mutual SIFT ratio matches and fundamental-matrix RANSAC."""

from pathlib import Path

import cv2
import numpy as np


def match_descriptors(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.8) -> list:
    if min(len(desc1), len(desc2)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(desc1, desc2, k=2)
    backward = matcher.knnMatch(desc2, desc1, k=2)
    reverse = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < ratio * n.distance}
    return [m for m, n in forward if m.distance < ratio * n.distance and reverse.get(m.trainIdx) == m.queryIdx]


def build_pair_matches(descriptors: list[np.ndarray], keypoints: list, max_gap: int = 6) -> dict:
    count = len(descriptors)
    pairs = sorted({tuple(sorted((i, (i + gap) % count))) for i in range(count)
                    for gap in range(1, min(max_gap + 1, count))})
    pair_matches = {}
    for index, (i, j) in enumerate(pairs):
        matches = match_descriptors(descriptors[i], descriptors[j])
        if len(matches) >= 15:
            xy1 = np.float64([keypoints[i][m.queryIdx].pt for m in matches])
            xy2 = np.float64([keypoints[j][m.trainIdx].pt for m in matches])
            F, mask = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, 1.5, 0.999)
            if F is not None and mask is not None:
                pair_matches[(i, j)] = [m for m, ok in zip(matches, mask.ravel()) if ok]
        if (index + 1) % 50 == 0 or index + 1 == len(pairs):
            print(f"Matching {index + 1}/{len(pairs)} pairs", flush=True)
    return pair_matches


def oriented_matches(pair_matches: dict, i: int, j: int) -> list[tuple[int, int]]:
    matches = pair_matches.get((min(i, j), max(i, j)), [])
    if i < j:
        return [(m.queryIdx, m.trainIdx) for m in matches]
    return [(m.trainIdx, m.queryIdx) for m in matches]


def save_match_visualization(images: list, keypoints: list, pair_matches: dict, pair: tuple, path: Path) -> None:
    i, j = pair
    matches = sorted(pair_matches[pair], key=lambda m: m.distance)[:70]
    vis = cv2.drawMatches(images[i], keypoints[i], images[j], keypoints[j], matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), vis):
        raise OSError(f"Cannot write {path}")
