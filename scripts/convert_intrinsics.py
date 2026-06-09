#!/usr/bin/env python3
# convert_intrinsics.py
# Normalize intrinsics .npz files to keys expected by manual_extrinsics.py:
#   camera_matrix (3x3), dist_coeffs (1xN), image_size (w,h)

import argparse
import sys
import numpy as np
from pathlib import Path

KEY_ALIASES = {
    "K": ["camera_matrix", "K", "cameraMatrix", "mtx"],
    "D": ["dist_coeffs", "D", "distCoeffs", "dist", "distortion_coefficients"],
    "SIZE": ["image_size", "img_size", "size", "resolution"],
}

def load_generic_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    # Support savez(dict) pattern (single arr_0 object-dtype)
    if list(z.files) == ["arr_0"]:
        arr0 = z["arr_0"]
        if isinstance(arr0, np.ndarray) and arr0.dtype == object and arr0.size == 1:
            obj = arr0.item()
            if isinstance(obj, dict):
                return obj
    return {k: z[k] for k in z.files}

def pick(d: dict, *names):
    for n in names:
        if n in d:
            return d[n]
    raise KeyError(f"Missing any of {names} in source data.")

def normalize_K(K) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.size != 9:
        raise ValueError(f"camera_matrix must have 9 elements; got shape {K.shape}")
    return K.reshape(3, 3)

def normalize_D(D) -> np.ndarray:
    # OpenCV accepts 1xN or Nx1. We'll write 1xN to be consistent.
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    return D.reshape(1, -1)

def normalize_size(d: dict) -> tuple[int,int]:
    for k in KEY_ALIASES["SIZE"]:
        if k in d:
            s = np.array(d[k]).astype(int).reshape(-1)
            if s.size == 2:
                return (int(s[0]), int(s[1]))
    w = d.get("w") or d.get("width")
    h = d.get("h") or d.get("height")
    if w is not None and h is not None:
        return (int(w), int(h))
    # Last resort; your script doesn’t truly need sizes, but keep placeholders.
    return (0, 0)

def convert_one(in_path: Path, out_path: Path, quiet: bool=False):
    src = load_generic_npz(in_path)
    K_raw = pick(src, *KEY_ALIASES["K"])
    D_raw = pick(src, *KEY_ALIASES["D"])
    size  = normalize_size(src)

    K = normalize_K(K_raw)
    D = normalize_D(D_raw)

    np.savez(out_path, camera_matrix=K, dist_coeffs=D, image_size=size)
    if not quiet:
        print(f"[OK] {in_path.name}  ->  {out_path.name}   "
              f"(image_size={size}, D shape={D.shape})")

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert intrinsics NPZs to expected keys: camera_matrix, dist_coeffs, image_size"
    )
    ap.add_argument("inputs", nargs="+", help="Input .npz files")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: alongside input)")
    ap.add_argument("--suffix", default="_fixed", help="Suffix to append to filename stem (default: _fixed)")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-file logs")
    args = ap.parse_args(argv)

    rc = 0
    for ip in args.inputs:
        in_path = Path(ip)
        if not in_path.exists():
            print(f"[ERR] Missing file: {in_path}", file=sys.stderr)
            rc = 2
            continue
        out_dir = args.out_dir if args.out_dir else in_path.parent
        out_path = out_dir / f"{in_path.stem}{args.suffix}.npz"

        try:
            convert_one(in_path, out_path, quiet=args.quiet)
        except Exception as e:
            print(f"[ERR] {in_path.name}: {e}", file=sys.stderr)
            rc = 1
    return rc

if __name__ == "__main__":
    sys.exit(main())

