#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import struct
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

def load_colmap_points3d(bin_file: str) -> np.ndarray:
    """
    解析 COLMAP 的 points3D.bin，回傳 Nx3 的 XYZ。
    只取座標（忽略顏色/track 等），以降低記憶體與實作相似度。
    """
    with open(bin_file, "rb") as f:
        blob = f.read()

    pos = 0
    def eat(fmt: str):
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, blob, pos)
        pos += size
        return vals if len(vals) > 1 else vals[0]

    n_pts = eat("<Q")
    xyz_accum = np.empty((n_pts, 3), dtype=np.float64)

    for i in range(n_pts):
        _pid = eat("<Q")
        x, y, z = eat("<3d")
        _r, _g, _b = eat("<3B")
        _err = eat("<d")
        track_len = eat("<Q")
        pos += track_len * struct.calcsize("<II")
        xyz_accum[i, :] = (x, y, z)

    return xyz_accum



def build_pcd(xyz: np.ndarray, voxel: Optional[float]) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if voxel and voxel > 0:
        p = p.voxel_down_sample(voxel)
    radius = voxel * 3.0 if voxel and voxel > 0 else 0.05
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    p.orient_normals_consistent_tangent_plane(30)
    return p


def tidy_mesh(m: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_non_manifold_edges()
    return m

class MeshBuilder:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd

    def reconstruct_poisson(
        self,
        depth: int,
        density_trim: float,
        target_tris: int,
        smooth_iter: int,
    ) -> o3d.geometry.TriangleMesh:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.pcd, depth=depth
        )
        if 0.0 < density_trim < 0.5:
            dens = np.asarray(densities)
            keep_mask = dens > np.quantile(dens, density_trim)
            mesh = mesh.select_by_index(np.where(keep_mask)[0])

        mesh = tidy_mesh(mesh)

        if smooth_iter and smooth_iter > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iter))

        if target_tris and target_tris > 0 and len(mesh.triangles) > target_tris:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=int(target_tris)
            )

        mesh.compute_vertex_normals()
        return mesh

    def reconstruct_bpa(
        self,
        radius: float,
        target_tris: int,
        smooth_iter: int,
    ) -> o3d.geometry.TriangleMesh:

        rset = o3d.utility.DoubleVector([radius, radius * 2.0, radius * 4.0])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            self.pcd, rset
        )
        mesh = tidy_mesh(mesh)

        if smooth_iter and smooth_iter > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iter))

        if target_tris and target_tris > 0 and len(mesh.triangles) > target_tris:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=int(target_tris)
            )

        mesh.compute_vertex_normals()
        return mesh


def preview(
    geoms: Tuple[o3d.geometry.Geometry, ...],
    title: str = "Q1-2 Mesh Preview",
    size: Tuple[int, int] = (1920, 1080),
    point_size: float = 3.0,
    bg=(1.0, 1.0, 1.0),
    save_png: Optional[str] = None,
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=size[0], height=size[1])
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg, dtype=np.float64)
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True

    vis.poll_events(); vis.update_renderer()
    if save_png:
        vis.capture_screen_image(save_png, do_render=True)
        print(f"[OK] screenshot -> {save_png}")
    vis.run()
    vis.destroy_window()


def parse_args():
    ap = argparse.ArgumentParser(description="COLMAP -> Open3D mesh reconstruction (alt impl.)")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--colmap_model_dir", type=str,
                    help="Directory containing points3D.bin (e.g., ./sparse/0)")
    src.add_argument("--input_ply", type=str,
                    help="Path to existing point cloud .ply file")

    ap.add_argument("--voxel", type=float, default=0.01,
                    help="Voxel downsampling size in meters before reconstruction. 0 means no downsampling.")
    ap.add_argument("--method", type=str, choices=["poisson", "bpa"], default="poisson")

    ap.add_argument("--poisson_depth", type=int, default=10,
                    help="Poisson octree depth (commonly 8–12)")
    ap.add_argument("--density_trim", type=float, default=0.05,
                    help="Density quantile threshold for trimming thin surfaces (0–0.5)")

    ap.add_argument("--bpa_radius", type=float, default=0.02,
                    help="BPA sphere radius (adjust based on scene scale)")

    ap.add_argument("--target_tris", type=int, default=300000,
                    help="Target number of triangles after simplification (0 means no simplification)")
    ap.add_argument("--out", type=str, default="mesh_out.ply",
                    help="Output mesh file path (.ply/.obj/.stl)")
    ap.add_argument("--no_view", action="store_true", help="Disable visualization preview")
    return ap.parse_args()


def main():
    args = parse_args()

    # 來源：COLMAP model 或既有 PLY
    if args.colmap_model_dir:
        pts_bin = os.path.join(args.colmap_model_dir, "points3D.bin")
        if not os.path.isfile(pts_bin):
            raise FileNotFoundError(f"points3D.bin not found: {pts_bin}")
        xyz = load_colmap_points3d(pts_bin)
        pcd = build_pcd(xyz, voxel=args.voxel if args.voxel > 0 else None)
    else:
        if not os.path.isfile(args.input_ply):
            raise FileNotFoundError(args.input_ply)
        pcd = o3d.io.read_point_cloud(args.input_ply)
        if not pcd.has_normals():
            r = args.voxel * 3.0 if args.voxel and args.voxel > 0 else 0.05
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(30)

    builder = MeshBuilder(pcd)

    if args.method == "poisson":
        mesh = builder.reconstruct_poisson(
            depth=args.poisson_depth,
            density_trim=args.density_trim,
            target_tris=args.target_tris,
            smooth_iter=3
        )
    else:
        mesh = builder.reconstruct_bpa(
            radius=args.bpa_radius,
            target_tris=args.target_tris,
            smooth_iter=1
        )

    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"[OK] Mesh saved: {args.out} | triangles={len(mesh.triangles)}")

    if not args.no_view:
        # Just for visualization.
        mesh.paint_uniform_color([0.82, 0.82, 0.82])
        pcd.paint_uniform_color([0.2, 0.6, 1.0])
        preview((pcd, mesh), title="Q1-2 Mesh Preview")


if __name__ == "__main__":
    main()

