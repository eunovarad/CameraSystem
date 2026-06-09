
# 📷 CameraSystem — Multi-Camera 3D Measurement Pipeline

## Overview

This repository implements a **multi-camera 3D measurement pipeline** using OpenCV.  
The system calibrates individual cameras, determines their spatial relationship (rig), and reconstructs 3D points from synchronized image captures. It then evaluates measurement accuracy against known phantom geometry.

The pipeline consists of four main stages:

1. Camera intrinsics calibration  
2. Multi-camera image capture  
3. Camera rig (extrinsics) estimation  
4. 3D reconstruction and accuracy validation  

---

## 🚀 Pipeline Summary

### 1. Capture Calibration Images

**Script:** `capture_until_q.py`

**Output:**
captures_left/.png
captures_right/.png
captures_back/*.png

Captures images from a single camera at regular intervals while you move a checkerboard through different positions and orientations.

---

### 2. Calibrate Camera Intrinsics

**Script:** `calibrate_intrinsics_v5.py`

**Input:**
- Calibration images

**Output:**

intrin_.npz
calib_overlays/*.png (optional)

Computes:
- Camera matrix (K)
- Distortion coefficients (D)

---

### 3. Audit Intrinsics

**Script:** `audit_intrinsics_v3.py`

**Input:**
- Intrinsics file
- Calibration images

**Output:**

*_audit.png (optional)
audit.csv (optional)

Validates calibration quality using reprojection error.

---

### 4. Capture Multi-Camera Data

**Script:** `3cam_simul_space.py`

**Output:**

phantom_images/
cam0_.png
cam1_.png
cam2_.png

Captures synchronized images from all cameras using a spacebar trigger.

---

### 5. Estimate Camera Rig (Extrinsics)

**Script:** `manual_extensics.py`

**Input:**
- Phantom image sets
- Intrinsics files

**Output:**

rig1.json
rig2.json
...

Uses manually selected correspondences to compute relative camera positions.

---

### 6. Aggregate Rigs

**Script:** `aggregate_rigs.py`

**Input:**
- Multiple rig estimates

**Output:**

rig_final.json
rig_final.npz

Combines multiple rig estimates into a single robust solution.

---

### 7. Validate System Accuracy

**Script:** `phantom_accuracy_test_auto.py`

**Input:**
- Final rig
- Intrinsics
- Phantom images

**Output:**

phantom_accuracy.csv

Reconstructs 3D points and compares them against known phantom geometry.

---

## 🧠 Detailed Script Descriptions

---

### 📌 capture_until_q.py

Captures images from a single camera for calibration.

- Uses a threaded camera stream (`CameraStream`)
- Saves full-resolution images at fixed intervals
- Requires moving the checkerboard through different positions and angles

**Key behavior:**
- Continuous capture loop
- Timestamped image saving
- Scaled live preview

---

### 📌 cam_stream.py

Threaded camera capture utility used across all capture scripts.

- Runs image acquisition in a background thread
- Provides the most recent frame on demand
- Improves performance and synchronization for multi-camera use

**Key features:**
- Resolution and FPS control
- Thread-safe frame access
- Multi-camera capable

---

### 📌 calibrate_intrinsics_v5.py

Performs camera calibration using checkerboard or Charuco patterns.

**What it does:**
- Detects calibration pattern corners
- Matches 2D image points to known 3D geometry
- Solves for camera intrinsics using OpenCV

**Outputs:**
- `.npz` file containing:
  - Camera matrix (K)
  - Distortion coefficients (D)
  - RMS reprojection error

---

### 📌 audit_intrinsics_v3.py

Validates calibration results.

**What it does:**
- Re-detects corners
- Reprojects known 3D points using the camera model
- Compares predicted vs actual image points

**Outputs:**
- RMS error per image
- Optional overlay images for visualization

---

### 📌 3cam_simul_space.py

Captures synchronized images from multiple cameras.

**What it does:**
- Opens all cameras simultaneously
- Displays live feeds
- Saves images with a shared timestamp when spacebar is pressed

**Important:**
- Each capture set represents the same moment in time
- Required for 3D reconstruction

---

### 📌 manual_extensics.py

Computes camera extrinsics (rig geometry).

**What it does:**
- Loads synchronized image sets
- Requires manual selection of corresponding phantom points
- Uses known phantom geometry with `solvePnP`

**Outputs:**
- Camera transformation matrices
- One rig estimate per capture set

---

### 📌 aggregate_rigs.py

Combines multiple rig estimates.

**What it does:**
- Filters out poor-quality rigs
- Averages rotations (quaternion mean)
- Uses median translation for robustness

**Outputs:**
- `rig_final.json`
- `rig_final.npz`

---

### 📌 phantom_accuracy_test_auto.py

Evaluates system accuracy.

**What it does:**
- Triangulates 3D points from multiple camera views
- Compares reconstructed points to reference phantom coordinates

**Outputs:**
- Per-point error metrics
- CSV summary of reconstruction accuracy

---

## 📊 Output Interpretation

### phantom_accuracy.csv

Each row represents a reconstructed phantom point.

| Field | Description |
|------|-------------|
| fid_id | Phantom feature ID |
| X_meas, Y_meas, Z_meas | Measured 3D position |
| X_ref, Y_ref, Z_ref | Reference 3D position |
| dX, dY, dZ | Error components |
| err_norm | 3D error magnitude (mm) |
| rms_px | Pixel reprojection error |

---

## ✅ Key Concepts

- **Intrinsics** → how each camera forms an image  
- **Extrinsics (Rig)** → spatial relationship between cameras  
- **Triangulation** → reconstructing 3D points from multiple views  
- **Reprojection error** → calibration accuracy metric  

---

## ⚠️ Notes

- Calibration quality directly impacts final accuracy  
- Phantom must be visible in all cameras  
- Use multiple capture positions for robustness  
- Large errors typically indicate mis-clicks or detection failures  

---

## 🎯 Typical Workflow



capture_until_q.py         → capture checkerboard images
calibrate_intrinsics       → compute intrinsics
audit_intrinsics           → verify calibration
3cam_simul_space           → capture phantom images
manual_extensics           → compute rig