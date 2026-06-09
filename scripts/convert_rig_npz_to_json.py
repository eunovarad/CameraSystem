
#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

def npz_to_plain_dict(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    d = {}
    for k in z.files:
        v = z[k]
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                try:
                    v = v.item()
                except Exception:
                    v = v.tolist()
            else:
                v = v.tolist()
        d[k] = v
    return d

def build_output(rig_dict: dict, template: dict|None) -> dict:
    out = {}
    # frames
    if "frames" in rig_dict and isinstance(rig_dict["frames"], dict):
        out["frames"] = rig_dict["frames"]
    else:
        out["frames"] = (template.get("frames") if template else {"reference": "cam0"})
    # images: only include if present in rig data
    if "images" in rig_dict and isinstance(rig_dict["images"], (list, tuple)):
        out["images"] = list(rig_dict["images"])
    # intrinsics mapping if present
    if "intrinsics" in rig_dict and isinstance(rig_dict["intrinsics"], dict):
        out["intrinsics"] = rig_dict["intrinsics"]
    # transforms and metadata
    for k in ["H_cam0","H_cam1","H_cam2","H_cam1_in_cam0","H_cam2_in_cam0","reprojection_rms_px","phantom_file"]:
        if k in rig_dict:
            out[k] = rig_dict[k]
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert rig .npz to .json (structure guided by an optional template JSON).")
    ap.add_argument("inputs", nargs="+", help="Input rig .npz files")
    ap.add_argument("--template", type=Path, default=None, help="Template JSON (e.g., rig_from_clicks.json)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: alongside input)")
    ap.add_argument("--suffix", default="", help="Suffix for output filename stem (default: none)")
    args = ap.parse_args()

    template = None
    if args.template:
        with open(args.template, "r") as f:
            template = json.load(f)

    rc = 0
    for ip in args.inputs:
        in_path = Path(ip)
        if not in_path.exists():
            print(f"[ERR] Missing file: {in_path}")
            rc = 2
            continue
        rig_dict = npz_to_plain_dict(in_path)
        out = build_output(rig_dict, template)
        out_dir = args.out_dir if args.out_dir else in_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (in_path.stem + args.suffix + ".json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] {in_path.name} -> {out_path.name}")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
