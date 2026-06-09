#!/usr/bin/env python3
import argparse, glob, json
import numpy as np

def average_rotations(Rs):
    # quaternion mean
    Q = []
    for R in Rs:
        t = np.trace(R)
        w = np.sqrt(max(0.0, 1.0 + t))/2.0
        if w < 1e-12: q = np.array([1,0,0,0],float)
        else:
            x=(R[2,1]-R[1,2])/(4*w); y=(R[0,2]-R[2,0])/(4*w); z=(R[1,0]-R[0,1])/(4*w)
            q=np.array([w,x,y,z],float)
        q/=np.linalg.norm(q); Q.append(q)
    q=np.mean(np.stack(Q,0),axis=0); q/=np.linalg.norm(q)
    w,x,y,z=q
    Rm=np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ],float)
    return Rm

def se3_errors(H_list, H_avg):
    angs=[]; dists=[]
    for H in H_list:
        dH = np.linalg.inv(H_avg) @ H
        R = dH[:3,:3]; t = dH[:3,3]
        ang = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2.0, -1.0, 1.0)))
        angs.append(ang); dists.append(np.linalg.norm(t))
    return np.array(angs), np.array(dists)

def main():
    ap = argparse.ArgumentParser(description="Aggregate multiple manual rig json files")
    ap.add_argument("json_files", nargs="+", help="rig_from_clicks_*.json files")
    ap.add_argument("--json_out", default="rig_final.json")
    ap.add_argument("--npz_out", default="rig_final.npz")
    args = ap.parse_args()

    H01_list=[]; H02_list=[]
    for p in args.json_files:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        H01_list.append(np.array(d["H_cam1_in_cam0"], dtype=float))
        H02_list.append(np.array(d["H_cam2_in_cam0"], dtype=float))

    R01s=[H[:3,:3] for H in H01_list]; t01s=np.array([H[:3,3] for H in H01_list])
    R02s=[H[:3,:3] for H in H02_list]; t02s=np.array([H[:3,3] for H in H02_list])

    R01=average_rotations(R01s); t01=np.median(t01s,axis=0)
    R02=average_rotations(R02s); t02=np.median(t02s,axis=0)

    H01=np.eye(4); H01[:3,:3]=R01; H01[:3,3]=t01
    H02=np.eye(4); H02[:3,:3]=R02; H02[:3,3]=t02

    a01,d01=se3_errors(H01_list,H01)
    a02,d02=se3_errors(H02_list,H02)

    np.savez(args.npz_out, H_cam0=np.eye(4), H_cam1_in_cam0=H01, H_cam2_in_cam0=H02,
             H01_samples=np.array(H01_list), H02_samples=np.array(H02_list),
             rot_err01_deg=a01, rot_err02_deg=a02, trans_err01=np.array(t01s)-t01, trans_err02=np.array(t02s)-t02)
    print(f"Wrote {args.npz_out}")
    print(f"(0-1) rot spread: mean={a01.mean():.4f}°  std={a01.std():.4f}°; trans MAD≈{np.median(np.abs(t01s-t01),axis=0)}")
    print(f"(0-2) rot spread: mean={a02.mean():.4f}°  std={a02.std():.4f}°; trans MAD≈{np.median(np.abs(t02s-t02),axis=0)}")

    def tolist(M): return np.asarray(M,float).tolist()
    with open(args.json_out,"w",encoding="utf-8") as f:
        json.dump({
            "frames":{"reference":"cam0"},
            "H_cam1_in_cam0": tolist(H01),
            "H_cam2_in_cam0": tolist(H02),
            "spread_deg": {
                "cam1_in_cam0": {"mean": float(a01.mean()), "std": float(a01.std())},
                "cam2_in_cam0": {"mean": float(a02.mean()), "std": float(a02.std())}
            },
            "num_samples": {"H01": len(H01_list), "H02": len(H02_list)}
        }, f, indent=2)
    print(f"Wrote {args.json_out}")

if __name__=="__main__":
    main()