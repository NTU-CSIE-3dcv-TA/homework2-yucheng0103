#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse
import numpy as np
import pandas as pd
import cv2 as cv
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ------------------------- helpers -------------------------

def load_point_cloud(points3D_df):
    xyz = np.vstack(points3D_df["XYZ"])
    rgb = np.vstack(points3D_df["RGB"]) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(
        [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
    )
    axes.lines  = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]])
    axes.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
    return axes

def get_transform_mat(rotation_deg, translation, scale):
    Rm = R.from_euler("xyz", rotation_deg, degrees=True).as_matrix()
    Sm = np.eye(3) * float(scale)
    T  = np.concatenate([Sm @ Rm, translation.reshape(3,1)], axis=1)
    return T

def make_cube_points(res=50):
    """6 faces of a unit cube sampled on a grid, with per-face colors."""
    u = np.linspace(0.0, 1.0, res)
    uu, vv = np.meshgrid(u, u); uu = uu.reshape(-1,1); vv = vv.reshape(-1,1)
    faces, cols = [], []
    # z=0, z=1, y=0, y=1, x=0, x=1
    faces.append(np.hstack([uu, vv, np.zeros_like(uu)]));         cols.append([0,200,255])
    faces.append(np.hstack([uu, vv, np.ones_like(uu)]));          cols.append([0,120,240])
    faces.append(np.hstack([uu, np.zeros_like(uu), vv]));         cols.append([0,255,0])
    faces.append(np.hstack([uu, np.ones_like(uu),  vv]));         cols.append([255,0,0])
    faces.append(np.hstack([np.zeros_like(uu), uu, vv]));         cols.append([255,0,255])
    faces.append(np.hstack([np.ones_like(uu),  uu, vv]));         cols.append([0,0,255])
    pts = np.vstack(faces)
    clr = np.vstack([np.tile(np.array(c)/255.0, (res*res,1)) for c in cols]).astype(np.float32)
    return pts, clr

# --------------------- interactive part --------------------

# Globals used by key callbacks (same shape as the base code)
cube_pcd = None
cube_vertices = None
cube_colors = None
R_euler = None
t = None
scale = None
vis = None
shift_pressed = False

def update_cube():
    """Apply current transform to local cube points and refresh Open3D object."""
    global cube_pcd, cube_vertices, cube_colors, R_euler, t, scale, vis
    T = get_transform_mat(R_euler, t, scale)
    pts_h = np.hstack([cube_vertices, np.ones((cube_vertices.shape[0],1))])
    pts_w = (T @ pts_h.T).T[:, :3]
    cube_pcd.points = o3d.utility.Vector3dVector(pts_w)
    cube_pcd.colors = o3d.utility.Vector3dVector(cube_colors)
    if vis is not None:
        vis.update_geometry(cube_pcd)

def toggle_key_shift(_vis, action, _mods):
    global shift_pressed
    shift_pressed = (action == 1)   # 1: key down, 0: key up
    return True

def _step(val, small, big):
    return -small if shift_pressed else small if val == "small" else (-big if shift_pressed else big)

def update_tx(_):  # A / Shift+A
    global t; t[0] += _step("small", 0.01, 0.0); update_cube()
def update_ty(_):  # S / Shift+S
    global t; t[1] += _step("small", 0.01, 0.0); update_cube()
def update_tz(_):  # D / Shift+D
    global t; t[2] += _step("small", 0.01, 0.0); update_cube()
def update_rx(_):  # Z / Shift+Z
    global R_euler; R_euler[0] += _step("small", 1.0, 0.0); update_cube()
def update_ry(_):  # X / Shift+X
    global R_euler; R_euler[1] += _step("small", 1.0, 0.0); update_cube()
def update_rz(_):  # C / Shift+C
    global R_euler; R_euler[2] += _step("small", 1.0, 0.0); update_cube()
def update_scale(_):  # V / Shift+V
    global scale; scale += _step("small", 0.05, 0.0); update_cube()

# ----------------------- AR rendering ----------------------

def render_ar_video(T_cube, cube_local_pts, cube_colors_rgb,
                    images_df_path="data/images.pkl",
                    images_root="data/frames",
                    out_path="output.mp4",
                    fps=10):
    # intrinsics (same numbers)
    K = np.array([[1868.27, 0.0, 540.0],
                  [   0.0, 1869.18, 960.0],
                  [   0.0,    0.0,   1.0]], dtype=np.float64)
    DIST = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352, 0.0], dtype=np.float64)

    if not os.path.exists(images_df_path):
        raise FileNotFoundError(f"Missing file: {images_df_path}")
    df = pd.read_pickle(images_df_path)

    ones = np.ones((cube_local_pts.shape[0], 1))
    P_world = (T_cube @ np.hstack([cube_local_pts, ones]).T).T[:, :3]
    bgr = (cube_colors_rgb[:, ::-1] * 255.0).astype(np.uint8)

    frames, size_wh = [], None
    for _, row in df.iterrows():
        # locate image path
        name = (row.get("IMAGE_PATH") or row.get("NAME") or
                row.get("filename") or row.get("path"))
        if not isinstance(name, str): continue
        img_path = name if os.path.isabs(name) else os.path.join(images_root, name)

        img = cv.imread(img_path)
        if img is None: continue
        if size_wh is None: h, w = img.shape[:2]; size_wh = (w, h)

        if not all(k in row for k in ("QX","QY","QZ","QW","TX","TY","TZ")):
            frames.append(img); continue

        # world->camera (COLMAP convention)
        Rcw = R.from_quat([float(row["QX"]), float(row["QY"]), float(row["QZ"]), float(row["QW"])]).as_matrix()
        tcw = np.array([[float(row["TX"])],
                        [float(row["TY"])],
                        [float(row["TZ"])]], dtype=np.float64)

        # project with a simple painter's algorithm (back-to-front)
        Pw = P_world
        Pc = (Rcw @ Pw.T) + tcw
        Z = Pc[2, :]
        mask = Z > 0
        if not np.any(mask):
            frames.append(img); continue

        order = np.argsort(Z[mask])  # far -> near
        pts = Pw[mask][order]
        cols = bgr[mask][order]

        rvec, _ = cv.Rodrigues(Rcw)
        uv, _ = cv.projectPoints(pts, rvec, tcw, K, DIST)
        uv = uv.reshape(-1, 2).astype(np.int32)

        canvas = img.copy()
        H, W = canvas.shape[:2]
        for (x, y), c in zip(uv, cols):
            if 0 <= x < W and 0 <= y < H:
                cv.circle(canvas, (int(x), int(y)), 2, tuple(int(v) for v in c), -1, lineType=cv.LINE_AA)
        frames.append(canvas)

    if frames:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        vw = cv.VideoWriter(out_path, fourcc, fps, size_wh)
        for f in frames: vw.write(f)
        vw.release()
        print(f"[OK] wrote {out_path} with {len(frames)} frames.")
    else:
        print("[WARN] no frames produced; check paths in images.pkl.")

# --------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Interactive cube placement + AR rendering")
    ap.add_argument("--points3d", default="data/points3D.pkl")
    ap.add_argument("--images_df", default="data/images.pkl")
    ap.add_argument("--frames_root", default="data/frames")
    ap.add_argument("--video_out", default="output.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--cube_res", type=int, default=50)
    ap.add_argument("--no_view", action="store_true", help="skip Open3D window (use saved or identity transform)")
    args = ap.parse_args()

    # ---- interactive placement (unless --no_view) ----
    global cube_pcd, cube_vertices, cube_colors, R_euler, t, scale, vis

    # make colored cube as points
    cube_vertices, cube_colors = make_cube_points(res=args.cube_res)
    cube_pcd = o3d.geometry.PointCloud()
    cube_pcd.points = o3d.utility.Vector3dVector(cube_vertices)
    cube_pcd.colors = o3d.utility.Vector3dVector(cube_colors)

    # default transform
    R_euler = np.array([0.0, 0.0, 0.0])
    t       = np.array([0.0, 0.0, 0.0])
    scale   = 1.0

    if not args.no_view:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window():
            print("[WARN] Open3D could not open a window. Continue headless with identity transform.")
            vis = None
        else:
            # scene cloud + axes + cube
            scene_df = pd.read_pickle(args.points3d)
            vis.add_geometry(load_point_cloud(scene_df))
            vis.add_geometry(load_axes())
            vis.add_geometry(cube_pcd)
            update_cube()

            # set a reasonable initial camera (same as base)
            vc = vis.get_view_control()
            cp = vc.convert_to_pinhole_camera_parameters()
            init = get_transform_mat(np.array([7.227, -16.950, -14.868]),
                                     np.array([-0.351, 1.036, 5.132]), 1.0)
            init = np.vstack([init, [0,0,0,1]])
            setattr(cp, "extrinsic", init)
            vc.convert_from_pinhole_camera_parameters(cp)

            # key bindings (same as base)
            vis.register_key_action_callback(340, toggle_key_shift)   # Shift (left)
            vis.register_key_action_callback(344, toggle_key_shift)   # Shift (right)
            vis.register_key_callback(ord('A'), update_tx)
            vis.register_key_callback(ord('S'), update_ty)
            vis.register_key_callback(ord('D'), update_tz)
            vis.register_key_callback(ord('Z'), update_rx)
            vis.register_key_callback(ord('X'), update_ry)
            vis.register_key_callback(ord('C'), update_rz)
            vis.register_key_callback(ord('V'), update_scale)

            print('[Keyboard usage]')
            print('Translate X: A / Shift+A')
            print('Translate Y: S / Shift+S')
            print('Translate Z: D / Shift+D')
            print('Rotate    X: Z / Shift+Z')
            print('Rotate    Y: X / Shift+X')
            print('Rotate    Z: C / Shift+C')
            print('Scale       : V / Shift+V')

            vis.run()
            vis.destroy_window()

    # Save artifacts (compatible names)
    T_cube = get_transform_mat(R_euler, t, scale)
    np.save("cube_transform_mat.npy", T_cube)
    np.save("cube_points.npy",        np.asarray(cube_pcd.points) if cube_pcd is not None else cube_vertices)
    np.save("cube_colors.npy",        np.asarray(cube_pcd.colors) if cube_pcd is not None else cube_colors)

    # ---- AR video rendering ----
    T_cube = np.load("cube_transform_mat.npy")
    pts    = np.load("cube_points.npy")
    cols   = np.load("cube_colors.npy")
    render_ar_video(T_cube, pts, cols,
                    images_df_path=args.images_df,
                    images_root=args.frames_root,
                    out_path=args.video_out,
                    fps=args.fps)

if __name__ == "__main__":
    main()
