#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random, argparse, cv2, os
from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428)    # do not change this seed

# ---------- helpers ----------
def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack).apply(average).reset_index()
    return desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")

K_DEF = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]], dtype=np.float64)
D_DEF = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352], dtype=np.float64)

# ---------- OpenCV baseline PnP ----------
def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    K, D = K_DEF, D_DEF

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pairs = bf.knnMatch(desc_query, desc_model, k=2)
    good = [m for m,n in pairs if m.distance < 0.75*n.distance]
    if len(good) < 6: return False, None, None, None

    pts2d = np.float32([kp_query[m.queryIdx] for m in good]).reshape(-1,1,2)
    pts3d = np.float32([kp_model[m.trainIdx] for m in good]).reshape(-1,1,3)

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        pts3d, pts2d, K, D,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=6.0, iterationsCount=300, confidence=0.999
    )
    if not ok or inl is None or len(inl) < 6: return False, None, None, None

    in2d, in3d = pts2d[inl.ravel()], pts3d[inl.ravel()]
    ok, rvec, tvec = cv2.solvePnP(in3d, in2d, K, D, rvec, tvec,
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    return ok, rvec, tvec, inl

# ---------- Original P3P + RANSAC (pixel-domain, no refine) ----------
def _p3p_numeric(Pw, uv, K, max_iter=30, tol=1e-8):
    uv1 = np.hstack([uv, np.ones((3,1))])
    f = (np.linalg.inv(K) @ uv1.T).T
    f /= np.linalg.norm(f, axis=1, keepdims=True)

    a, b, c = np.linalg.norm(Pw[1]-Pw[2]), np.linalg.norm(Pw[0]-Pw[2]), np.linalg.norm(Pw[0]-Pw[1])
    a2, b2, c2 = a*a, b*b, c*c

    def res(l):
        P1, P2, P3 = l[0]*f[0], l[1]*f[1], l[2]*f[2]
        return np.array([
            np.dot(P1-P2, P1-P2) - c2,
            np.dot(P1-P3, P1-P3) - b2,
            np.dot(P2-P3, P2-P3) - a2
        ])

    def jac(l):
        P1, P2, P3 = l[0]*f[0], l[1]*f[1], l[2]*f[2]
        J = np.zeros((3,3))
        J[0,0], J[0,1] = 2*np.dot(P1-P2, f[0]), -2*np.dot(P1-P2, f[1])
        J[1,0], J[1,2] = 2*np.dot(P1-P3, f[0]), -2*np.dot(P1-P3, f[2])
        J[2,1], J[2,2] = 2*np.dot(P2-P3, f[1]), -2*np.dot(P2-P3, f[2])
        return J

    base = (a+b+c)/3.0
    sols, seen = [], set()
    for s in [0.5, 1.0, 1.5, 2.0]:
        l, mu = np.array([base*s]*3, float), 1e-2
        for _ in range(max_iter):
            r, J = res(l), jac(l)
            try: d = np.linalg.solve(J.T@J + mu*np.eye(3), -J.T@r)
            except np.linalg.LinAlgError: break
            l_new = l + d
            if np.linalg.norm(res(l_new)) < np.linalg.norm(r):
                l, mu = l_new, mu*0.5
                if np.linalg.norm(d) < tol: break
            else: mu *= 2.0
        if np.any(l <= 0): continue

        Pc = (l[:,None]*f)
        Pw_c, Pc_c = Pw.mean(0), Pc.mean(0)
        X, Y = Pw - Pw_c, Pc - Pc_c
        U,S,Vt = np.linalg.svd(X.T @ Y)
        Rc = Vt.T @ U.T
        if np.linalg.det(Rc) < 0: Vt[-1]*=-1; Rc = Vt.T @ U.T
        tc = (Pc_c - Rc @ Pw_c).reshape(3,1)
        if np.all((Rc @ Pw.T + tc).T[:,2] > 0):
            key = tuple(np.round(np.hstack([Rc.ravel(), tc.ravel()]), 6))
            if key not in seen:
                seen.add(key); sols.append((Rc, tc))
    return sols

def _project_pix(K, Rc, tc, Pw):
    Pc = Pw @ Rc.T + tc.ravel()
    z = np.clip(Pc[:,2], 1e-12, None)
    u = K[0,0]*Pc[:,0]/z + K[0,2] + K[0,1]*Pc[:,1]/z
    v = K[1,1]*Pc[:,1]/z + K[1,2]
    return np.column_stack([u,v]), z

def ransac_p3p(query, model, thresh_px=4.0, max_iter=2000, confidence=0.999):
    kp_q, desc_q = query; kp_m, desc_m = model
    K = K_DEF

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pairs = bf.knnMatch(desc_q, desc_m, k=2)
    good = [m for m,n in pairs if m.distance < 0.75*n.distance]
    if len(good) < 6: return False, None, None, np.array([], int)

    uv = np.float64([kp_q[m.queryIdx] for m in good])
    Pw = np.float64([kp_m[m.trainIdx] for m in good])
    N = len(Pw)

    best_inl, best_model = None, None
    trials, max_trials = 0, max_iter
    rng = np.random.default_rng(1428)

    while trials < max_trials:
        idx = rng.choice(N, 3, replace=False)
        for Rc, tc in _p3p_numeric(Pw[idx], uv[idx], K):
            uv_hat, z = _project_pix(K, Rc, tc, Pw)
            err = np.linalg.norm(uv_hat - uv, axis=1)
            inl = np.where((err < thresh_px) & (z > 0))[0]
            if best_inl is None or len(inl) > len(best_inl):
                best_inl, best_model = inl, (Rc.copy(), tc.copy())
                w = max(1e-9, len(inl)/N)
                eps = min(max(1 - w**3, 1e-12), 1-1e-12)
                max_trials = min(max_iter, int(np.log(1-confidence)/np.log(eps))+1)
        trials += 1

    if best_model is None or len(best_inl) < 3:
        return False, None, None, np.array([], int)
    Rc, tc = best_model
    rvec = cv2.Rodrigues(Rc)[0]; tvec = tc
    return True, rvec, tvec, best_inl

# ---------- NEW: P3P + undistort + refine ----------
def _project_norm(Rc, tc, Pw):
    Pc = Pw @ Rc.T + tc.ravel()
    z = np.clip(Pc[:,2], 1e-12, None)
    return np.column_stack([Pc[:,0]/z, Pc[:,1]/z]), z

def ransac_p3p_refine(query, model, thresh_norm=0.004, max_iter=2000, confidence=0.999):
    kp_q, desc_q = query; kp_m, desc_m = model
    K, D = K_DEF, D_DEF

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pairs = bf.knnMatch(desc_q, desc_m, k=2)
    good = [m for m,n in pairs if m.distance < 0.75*n.distance]
    if len(good) < 6: return False, None, None, np.array([], int)

    uv_pix = np.float64([kp_q[m.queryIdx] for m in good])
    Pw_all = np.float64([kp_m[m.trainIdx] for m in good])
    N = len(Pw_all)

    uv_norm = cv2.undistortPoints(uv_pix.reshape(-1,1,2), K, D).reshape(-1,2)
    K_I = np.eye(3)

    best_inl, best_model = None, None
    trials, max_trials = 0, max_iter
    rng = np.random.default_rng(1428)

    while trials < max_trials:
        idx = rng.choice(N, 3, replace=False)
        for Rc, tc in _p3p_numeric(Pw_all[idx], uv_norm[idx], K_I):
            uv_hat, z = _project_norm(Rc, tc, Pw_all)
            err = np.linalg.norm(uv_hat - uv_norm, axis=1)
            inl = np.where((err < thresh_norm) & (z > 0))[0]
            if best_inl is None or len(inl) > len(best_inl):
                best_inl, best_model = inl, (Rc.copy(), tc.copy())
                w = max(1e-9, len(inl)/N)
                eps = min(max(1 - w**3, 1e-12), 1-1e-12)
                max_trials = min(max_iter, int(np.log(1-confidence)/np.log(eps))+1)
        trials += 1

    if best_model is None or len(best_inl) < 3:
        return False, None, None, np.array([], int)

    # refine on pixel domain with full intrinsics + distortion
    Rc, tc = best_model
    rvec0 = cv2.Rodrigues(Rc)[0]; tvec0 = tc.astype(np.float64)
    in3d = Pw_all[best_inl].reshape(-1,1,3).astype(np.float32)
    in2d = uv_pix[best_inl].reshape(-1,1,2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(in3d, in2d, K, D, rvec0, tvec0,
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: rvec, tvec = rvec0, tvec0
    return True, rvec, tvec, best_inl

# ---------- metrics ----------
def rotation_error(R1, R2):
    Rgt  = R.from_quat(R1).as_matrix()[0]
    Rest = R.from_quat(R2).as_matrix()[0]
    Rrel = Rgt.T @ Rest
    cos_theta = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def translation_error(t1, t2):
    return float(np.linalg.norm(t1 - t2))

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    import open3d as o3d
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    cam = np.array([
        [0,0,0], [-1,-0.75,-2], [1,-0.75,-2], [1,0.75,-2], [-1,0.75,-2]
    ], float) * 0.8
    edges = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]], int)

    geoms, traj = [pcd], []
    for c2w in Camera2World_Transform_Matrixs:
        hw = np.c_[cam, np.ones((cam.shape[0],1))]
        fw = (c2w @ hw.T).T[:,:3]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(fw)
        ls.lines  = o3d.utility.Vector2iVector(edges)
        ls.colors = o3d.utility.Vector3dVector(np.tile([[1,0,0]], (len(edges),1)))
        geoms.append(ls); traj.append(fw[0])

    if len(traj) >= 2:
        traj = np.asarray(traj)
        tls = o3d.geometry.LineSet()
        tls.points = o3d.utility.Vector3dVector(traj)
        tls.lines  = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(traj)-1)])
        tls.colors = o3d.utility.Vector3dVector(np.tile([[0,1,0]], (len(traj)-1,1)))
        geoms.append(tls)

    o3d.visualization.draw_geometries(geoms, window_name="Q2-1 Trajectory + PointCloud")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["opencv","p3p","p3p_refine"], default="opencv")
    ap.add_argument("--ids", type=str, default="200,201", help="comma-separated IMAGE_ID list")
    ap.add_argument("--no_view", action="store_true", help="disable Open3D window")
    args = ap.parse_args()
    image_ids = [int(x) for x in args.ids.split(",")]

    # Load data
    images_df   = pd.read_pickle("data/images.pkl")
    train_df    = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    r_list, t_list, rot_errs, trans_errs = [], [], [], []

    for idx in tqdm(image_ids):
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        # rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)  # not strictly needed for PnP here

        pts = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query  = np.array(pts["XY"].to_list())
        desc_query= np.array(pts["DESCRIPTORS"].to_list()).astype(np.float32)

        if args.solver == "opencv":
            ok, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        elif args.solver == "p3p":
            ok, rvec, tvec, inliers = ransac_p3p((kp_query, desc_query), (kp_model, desc_model))
        else:  # p3p_refine
            ok, rvec, tvec, inliers = ransac_p3p_refine((kp_query, desc_query), (kp_model, desc_model))

        r_list.append(rvec); t_list.append(tvec)

        gt = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = gt[["QX","QY","QZ","QW"]].values
        tvec_gt = gt[["TX","TY","TZ"]].values

        if ok and rvec is not None and tvec is not None:
            rotq_est = R.from_rotvec(rvec.reshape(1,3)).as_quat()
            tvec_est = tvec.reshape(1,3)
            rot_errs.append(rotation_error(rotq_gt, rotq_est))
            trans_errs.append(translation_error(tvec_gt, tvec_est))
        else:
            rot_errs.append(np.nan); trans_errs.append(np.nan)

    print(f"Median rotation error (deg): {np.nanmedian(rot_errs):.6f}")
    print(f"Median translation error (m): {np.nanmedian(trans_errs):.6f}")

    # Camera->World transforms for visualization
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        if (r is None) or (t is None): continue
        Rcw, _ = cv2.Rodrigues(r)  # world->camera
        tcw = t.reshape(3,1)
        Rwc = Rcw.T
        twc = -Rwc @ tcw
        c2w = np.eye(4)
        c2w[:3,:3] = Rwc
        c2w[:3, 3] = twc.ravel()
        Camera2World_Transform_Matrixs.append(c2w)

    visualization(Camera2World_Transform_Matrixs, points3D_df)


if __name__ == "__main__":
    main()
