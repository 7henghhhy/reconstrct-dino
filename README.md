# Dinosaur reconstruction

A computer vision course project: reconstruct a dinosaur exhibit from 62 calibrated photographs using incremental Structure from Motion (SfM).

![Reconstructed dinosaur exhibit and interactive viewer](docs/assets/preview.png)

**[Open the interactive demo](https://7henghhhy.github.io/reconstrct-dino/)** · [Download the dataset](https://github.com/7henghhhy/reconstrct-dino/releases/tag/dinosaur-data-v1)

## Run the demo

```bash
python3 -m http.server 8000 --directory docs
```

Open [localhost:8000](http://localhost:8000). Drag to rotate, shift-drag to pan, and scroll to zoom. The sidebar switches reconstruction stages, shows cameras, and selects input images. Point clouds and poses can be downloaded from the page.

## Run the reconstruction

Requires Python 3.11, OpenCV, NumPy, SciPy, and Open3D.

```bash
git clone https://github.com/7henghhhy/reconstrct-dino.git
cd reconstrct-dino
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and extract the [dataset](https://github.com/7henghhhy/reconstrct-dino/releases/tag/dinosaur-data-v1) at the repository root, or use GitHub CLI:

```bash
gh release download dinosaur-data-v1 --pattern dinosaur-data.zip
python -m zipfile -e dinosaur-data.zip .
```

The extracted folder should contain `重建所需数据/images/` (62 JPEGs) and `重建所需数据/camera_intrinsic.txt` (3×3 intrinsics).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python main.py --demo-dir docs/assets
```

Defaults: 1600×1200 images with scaled intrinsics, up to 6,000 SIFT features per image, and circular matching over six neighboring views. See `python main.py --help` for options.

### Outputs

| File in `outputs/` | Contents |
| --- | --- |
| `keypoints.jpg`, `matches.jpg` | Feature and match visualizations |
| `initial_points.ply`, `initialization.json` | Initial cloud and poses |
| `pre_ba.npz`, `reconstruction.npz` | Geometry and observations before/after BA |
| `final_points.ply`, `final_poses.txt` | Colored point cloud and world-to-camera poses |
| `reconstruction.ply`, `reconstruction.txt` | Submission copies; rename with `--student-id YOUR_ID` |
| `metrics.json` | Image order, errors, registration log, and solver status |

Each pose row is a flattened 4×4 world-to-camera matrix (62 rows for this dataset). `DJI_20200223_163225_243.jpg` is the reference camera: its pose is identity and appears first. Custom datasets fall back to the first sorted image. Missing registrations stop the run and are listed in `failure.json`.

## Method

SIFT → mutual ratio matching and RANSAC → essential-matrix initialization → PnP registration and triangulation → sparse bundle adjustment. The stages are split across `feature.py`, `match.py`, `initialize.py`, `reconstruct.py`, and `ba.py`, with `main.py` as the entry point.

New points need positive depth, at least 1.5° parallax, and at most 2 px reprojection error. PnP uses 3 px; final export filtering uses 4 px. BA uses SciPy sparse least squares with `soft_l1` loss and fixed intrinsics. The reference pose and one translation component of a second camera are fixed to remove gauge freedom.

## Results and ablations

| Configuration                  | Images | Points before export filtering | Mean error | Median error | Mean track length | Pair agreement | BA status            |
| ------------------------------ | ------ | ------------------------------ | ---------- | ------------ | ----------------- | -------------- | -------------------- |
| PnP, before BA                 | 62     | 33,385                         | 0.648 px   | 0.446 px     | 5.00              | —              | Disabled             |
| PnP + BA                       | 62     | 33,385                         | 0.345 px   | 0.263 px     | 5.00              | 99.9%          | ftol reached         |
| PnP + BA, half images          | 31     | 22,107                         | 0.327 px   | 0.259 px     | 3.34              | 98.6%          | Budget reached (300) |
| Essential-only, unit baselines | 62     | 64,629                         | 0.311 px   | 0.165 px     | 2.08              | 61.0%          | Disabled             |

The full run exports **33,383 points** and registers **62/62 cameras**. BA converges after 80 evaluations; the 31-image run reaches its 300-evaluation limit without converging.

Errors are measured at 1600×1200, before export filtering. The before/after BA rows use the same observations. Other runs use different tracks, so their errors are not directly comparable.

Pair agreement tests all verified matches for positive depth and ≤2 px error in both views, using the same matches for both 62-image pose methods.

```bash
# Without BA
python main.py --no-ba --output outputs/no_ba

# Every second image
python main.py --stride 2 --ba-evaluations 300 --output outputs/stride2

# Chained essential-matrix poses with unit baselines
python main.py --pose-method essential --no-ba --output outputs/essential
```

### Limits

This is a sparse reconstruction with arbitrary scale and limited surface coverage. No ground-truth poses or geometry were available. Low reprojection error alone does not establish 3D accuracy, particularly for the unit-baseline essential-only run. Intrinsics are fixed, distortion is ignored, and circular matching assumes this capture order; viewpoint gaps and repeated structures can break registration.

## Checks

```bash
python -m py_compile feature.py match.py initialize.py reconstruct.py ba.py main.py
python -m unittest discover -s tests -v
node --check docs/viewer.js
```
