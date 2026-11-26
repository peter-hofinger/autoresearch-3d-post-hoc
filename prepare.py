"""
One-time data preparation, sequential model caching, and fixed evaluation harness
for autoresearch-3d-post-hoc (Déjà View / DVLT + NVIDIA SAGE-10k on RTX 3080).

DO NOT MODIFY THIS FILE during autonomous experimentation.

Usage:
    uv run prepare.py                  # Full prep (28 SAGE-10k scenes + PBR render + DVLT K=8 + semantics)
    uv run prepare.py --pareto-ks      # Also cache DVLT for K in {4, 8, 12, 16} (Milestone 4 Pareto curve)
    uv run prepare.py --synthetic      # Fast zero-download synthetic benchmark (for 30s smoke test)
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    import open3d as o3d
except ImportError:
    o3d = None


# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

NUM_VIEWS = 10                  # Multi-view frames per scene
IMG_HEIGHT = 392                # 28 * 14 (exact DINOv2 / DVLT patch multiple)
IMG_WIDTH = 518                 # 37 * 14 (~4:3 aspect ratio)
TIME_BUDGET = 120               # Total post-hoc refinement budget across split (seconds)

DVLT_REPO_ID = "nvidia/dvlt"
DVLT_DEFAULT_K = 8              # Default Déjà View loop iterations
DVLT_PARETO_KS = (4, 8, 12, 16)

SAGE_REPO_ID = "nvidia/SAGE-10k"
SPLIT_SIZES = {"dev": 8, "val": 4, "test": 16}

# Fixed surface extraction & evaluation constants
TSDF_VOXEL_SIZE = 0.015         # 1.5 cm voxel grid
TSDF_SDF_TRUNC = 0.06           # 6.0 cm SDF truncation
TSDF_DEPTH_TRUNC = 10.0         # Max indoor depth (m)
FSCORE_THRESH = 0.02            # 2 cm F-score threshold
FSCORE_THRESH_LOOSE = 0.05      # 5 cm F-score threshold
NORMAL_MATCH_RADIUS = 0.05      # 5 cm radius for normal matching
BOUNDARY_DILATION_PX = 3        # Evaluation band around GT discontinuities / silhouettes
BOUNDARY_REL_DEPTH_JUMP = 0.06  # Relative depth jump threshold
MAX_EVAL_POINTS = 50_000        # Deterministic surface samples per scene
MIN_COVERAGE_RATIO = 0.95       # Guardrail: refined mask must retain >= 95% GT coverage

# Composite objective weights (lower val_score is better; raw DVLT baseline = 1.000000)
SCORE_WEIGHTS = {"cd": 0.30, "f1_err": 0.25, "nae": 0.25, "edge": 0.20}

# Semantic foundation models (cached one at a time for RTX 3080 10-12GB VRAM)
SAM2_MODEL_ID = "facebook/sam2.1-hiera-small"
SEG_MODEL_ID = "facebook/mask2former-swin-base-ade-semantic"
DINO_MODEL_ID = "facebook/dinov2-small"
DINO_PCA_DIM = 64

CACHE_DIR = Path(
    os.environ.get("AUTORESEARCH_3D_CACHE", os.path.expanduser("~/.cache/autoresearch-3d"))
)
RAW_SAGE_DIR = CACHE_DIR / "sage_raw"
SCENES_DIR = CACHE_DIR / "scenes"
SPLITS_PATH = CACHE_DIR / "splits.json"

# Structural taxonomy shared by SAGE-10k oracle & ADE20K Mask2Former:
# 0 = unknown/bg, 1 = planar_stuff, 2 = rigid_thing, 3 = thin_or_nonrigid
STRUCT_NAMES = {0: "unknown", 1: "planar_stuff", 2: "rigid_thing", 3: "thin_or_nonrigid"}

_PLANAR_KW = (
    "wall", "floor", "ceiling", "table", "desk", "counter", "shelf", "bookshelf", "door",
    "window", "whiteboard", "blackboard", "mirror", "carpet", "rug", "mat", "cabinet",
    "cupboard", "wardrobe", "dresser", "closet", "bed", "bench", "island", "panel",
    "board", "stair", "step", "tile", "ledge", "platform", "partition", "screen", "bar",
)
_THIN_KW = (
    "plant", "foliage", "leaf", "leaves", "flower", "tree", "bush", "fern", "curtain",
    "drape", "blind", "towel", "blanket", "pillow", "cushion", "cloth", "shirt", "jacket",
    "sheet", "duvet", "fabric", "cable", "wire", "cord", "rope", "person", "bag", "plush",
)


def classify_label_to_struct(label: str) -> int:
    name = str(label).lower().replace("-", " ").replace("_", " ").strip()
    if not name or name in ("none", "unknown", "void", "sky", "background", "unlabeled"):
        return 0
    if any(k in name for k in _THIN_KW):
        return 3
    if any(k in name for k in _PLANAR_KW):
        return 1
    return 2


# ---------------------------------------------------------------------------
# Deterministic Geometry, Alignment & Surface Extraction Utilities
# ---------------------------------------------------------------------------

def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Umeyama Sim(3): dst ≈ scale * (R @ src) + t."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = float(np.mean(np.sum(sc ** 2, axis=1)))
    if var_s < 1e-12:
        return 1.0, np.eye(3, dtype=np.float64), (mu_d - mu_s)
    U, D, Vt = np.linalg.svd((dc.T @ sc) / src.shape[0])
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    scale = max(float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0, 1e-6)
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def apply_sim3(depth: np.ndarray, poses: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen Sim(3) (scale, R, t) to depth (N,H,W) and c2w poses (N,4,4)."""
    d_out = (depth.astype(np.float64) * scale).astype(np.float32)
    p_out = np.zeros_like(poses, dtype=np.float64)
    p_out[:, 3, 3] = 1.0
    for i in range(poses.shape[0]):
        p_out[i, :3, :3] = R @ poses[i, :3, :3]
        p_out[i, :3, 3] = scale * (R @ poses[i, :3, 3]) + t
    return d_out, p_out.astype(np.float32)


def compute_depth_normals_np(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-space unit normals (H, W, 3) from depth (H, W) and intrinsics K (3, 3)."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([(u - K[0, 2]) * depth / max(K[0, 0], 1e-6), (v - K[1, 2]) * depth / max(K[1, 1], 1e-6), depth], axis=-1)
    dx = np.empty_like(pts)
    dy = np.empty_like(pts)
    dx[:, 1:-1], dx[:, 0], dx[:, -1] = 0.5 * (pts[:, 2:] - pts[:, :-2]), pts[:, 1] - pts[:, 0], pts[:, -1] - pts[:, -2]
    dy[1:-1, :], dy[0, :], dy[-1, :] = 0.5 * (pts[2:, :] - pts[:-2, :]), pts[1, :] - pts[0, :], pts[-1, :] - pts[-2, :]
    n = np.cross(dy, dx)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    n[n[..., 2] > 0] *= -1.0
    return n.astype(np.float32)


def compute_boundary_mask(gt_depth: np.ndarray, instance_id: np.ndarray | None = None) -> np.ndarray:
    """Exact (N, H, W) bool mask around GT depth discontinuities and instance silhouettes."""
    N, H, W = gt_depth.shape
    masks = np.zeros((N, H, W), dtype=bool)
    struct = ndimage.generate_binary_structure(2, 2)
    for i in range(N):
        z = gt_depth[i]
        valid = (z > 0.05) & (z < TSDF_DEPTH_TRUNC)
        dz = np.maximum(np.abs(np.diff(z, axis=1, prepend=z[:, :1])), np.abs(np.diff(z, axis=0, prepend=z[:1, :])))
        edge = valid & ((dz / np.maximum(z, 1e-3)) > BOUNDARY_REL_DEPTH_JUMP)
        if instance_id is not None:
            inst = instance_id[i]
            ix = inst != np.roll(inst, 1, axis=1)
            iy = inst != np.roll(inst, 1, axis=0)
            ix[:, 0] = False
            iy[0, :] = False
            edge |= valid & (ix | iy)
        if BOUNDARY_DILATION_PX > 0 and edge.any():
            edge = ndimage.binary_dilation(edge, structure=struct, iterations=BOUNDARY_DILATION_PX)
        masks[i] = edge & valid
    return masks


def unproject_voxel_surface(
    depths: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    valid_masks: np.ndarray | None = None,
    normals_world: np.ndarray | None = None,
    max_points: int = MAX_EVAL_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic back-projection + voxel averaging to (M, 3) points and unit normals."""
    N, H, W = depths.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts_list, nrm_list = [], []
    for i in range(N):
        z, K, c2w = depths[i], intrinsics[i], poses[i]
        m = np.isfinite(z) & (z > 0.05) & (z < TSDF_DEPTH_TRUNC)
        if valid_masks is not None:
            m &= valid_masks[i].astype(bool)
        if not np.any(m):
            continue
        pc = np.stack([(u[m] - K[0, 2]) * z[m] / max(K[0, 0], 1e-6), (v[m] - K[1, 2]) * z[m] / max(K[1, 1], 1e-6), z[m]], axis=-1)
        pts_list.append(pc @ c2w[:3, :3].T + c2w[:3, 3])
        nw = normals_world[i][m] if normals_world is not None else (compute_depth_normals_np(z, K)[m] @ c2w[:3, :3].T)
        nrm_list.append(nw)
    if not pts_list:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    pts, nrm = np.concatenate(pts_list, axis=0), np.concatenate(nrm_list, axis=0)
    vox = np.floor(pts / TSDF_VOXEL_SIZE).astype(np.int64)
    sh = vox - vox.min(axis=0)
    dims = sh.max(axis=0) + 1
    keys = sh[:, 0] * (dims[1] * dims[2]) + sh[:, 1] * dims[2] + sh[:, 2]
    order = np.argsort(keys, kind="mergesort")
    _, split_idx = np.unique(keys[order], return_index=True)
    counts = np.diff(np.append(split_idx, len(order))).astype(np.float32)[:, None]
    vp = np.add.reduceat(pts[order], split_idx, axis=0) / counts
    vn = np.add.reduceat(nrm[order], split_idx, axis=0)
    vn /= np.maximum(np.linalg.norm(vn, axis=-1, keepdims=True), 1e-8)

    if len(vp) > max_points:
        sel = np.sort(np.random.default_rng(42).choice(len(vp), size=max_points, replace=False))
        vp, vn = vp[sel], vn[sel]
    return vp.astype(np.float32), vn.astype(np.float32)


def extract_surface_tsdf(
    depths: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    valid_masks: np.ndarray | None = None,
    max_points: int = MAX_EVAL_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed Open3D ScalableTSDFVolume -> Marching Cubes -> area-weighted uniform surface samples."""
    if o3d is None:
        return unproject_voxel_surface(depths, poses, intrinsics, valid_masks, max_points=max_points)

    N, H, W = depths.shape
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL_SIZE,
        sdf_trunc=TSDF_SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(N):
        z = depths[i].astype(np.float32).copy()
        bad = (~np.isfinite(z)) | (z <= 0.05) | (z >= TSDF_DEPTH_TRUNC)
        if valid_masks is not None:
            bad |= ~valid_masks[i].astype(bool)
        z[bad] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(blank), o3d.geometry.Image(z),
            depth_scale=1.0, depth_trunc=TSDF_DEPTH_TRUNC, convert_rgb_to_intensity=False,
        )
        K = intrinsics[i]
        pin = o3d.camera.PinholeCameraIntrinsic(W, H, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        vol.integrate(rgbd, pin, np.linalg.inv(poses[i].astype(np.float64)))

    mesh = vol.extract_triangle_mesh()
    if len(mesh.triangles) < 100:
        return unproject_voxel_surface(depths, poses, intrinsics, valid_masks, max_points=max_points)

    mesh.compute_triangle_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    tnrm = np.asarray(mesh.triangle_normals, dtype=np.float64)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    if areas.sum() < 1e-8:
        return unproject_voxel_surface(depths, poses, intrinsics, valid_masks, max_points=max_points)

    rng = np.random.default_rng(42)
    tids = rng.choice(len(tris), size=max_points, p=areas / areas.sum())
    r1, r2 = np.sqrt(rng.random(max_points)), rng.random(max_points)
    pts = (1 - r1)[:, None] * v0[tids] + (r1 * (1 - r2))[:, None] * v1[tids] + (r1 * r2)[:, None] * v2[tids]
    nrm = tnrm[tids]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8)
    return pts.astype(np.float32), nrm.astype(np.float32)


def compute_fixed_sim3(
    pred_depth: np.ndarray, pred_poses: np.ndarray, pred_K: np.ndarray,
    gt_depth: np.ndarray, gt_poses: np.ndarray, gt_K: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute ONE frozen Sim(3) transform mapping raw DVLT frame -> GT world frame."""
    scale, R, t = umeyama_sim3(pred_poses[:, :3, 3], gt_poses[:, :3, 3], with_scale=True)
    valid = np.isfinite(pred_depth) & (pred_depth > 0.05) & np.isfinite(gt_depth) & (gt_depth > 0.05) & (gt_depth < TSDF_DEPTH_TRUNC)
    if np.sum(valid) > 1000:
        s_med = float(np.median(gt_depth[valid] / pred_depth[valid]))
        if not np.isfinite(scale) or scale <= 1e-4 or abs(math.log(max(scale / max(s_med, 1e-6), 1e-6))) > 0.5:
            scale = s_med
            t = gt_poses[:, :3, 3].mean(axis=0) - scale * (R @ pred_poses[:, :3, 3].mean(axis=0))

    # Trimmed Umeyama refinement on back-projected 3D correspondences (top 85% inliers)
    rng = np.random.default_rng(42)
    src_l, dst_l = [], []
    for i in range(gt_depth.shape[0]):
        ys, xs = np.where(valid[i])
        if len(xs) < 200:
            continue
        sel = rng.choice(len(xs), size=min(800, len(xs)), replace=False)
        xs, ys = xs[sel], ys[sel]
        kp, kg, zp, zg = pred_K[i], gt_K[i], pred_depth[i, ys, xs], gt_depth[i, ys, xs]
        pw = np.stack([(xs - kp[0, 2]) * zp / kp[0, 0], (ys - kp[1, 2]) * zp / kp[1, 1], zp], -1) @ pred_poses[i, :3, :3].T + pred_poses[i, :3, 3]
        gw = np.stack([(xs - kg[0, 2]) * zg / kg[0, 0], (ys - kg[1, 2]) * zg / kg[1, 1], zg], -1) @ gt_poses[i, :3, :3].T + gt_poses[i, :3, 3]
        src_l.append(pw)
        dst_l.append(gw)
    if src_l:
        src_a, dst_a = np.concatenate(src_l), np.concatenate(dst_l)
        errs = np.linalg.norm(scale * (src_a @ R.T) + t - dst_a, axis=-1)
        inl = errs <= np.percentile(errs, 85)
        if np.sum(inl) >= 50:
            scale, R, t = umeyama_sim3(src_a[inl], dst_a[inl], with_scale=True)
    return float(scale), R.astype(np.float64), t.astype(np.float64)


# ---------------------------------------------------------------------------
# Stage 0 & 1: Selective SAGE-10k Download & Headless PBR Rendering (bpy 4.3)
# ---------------------------------------------------------------------------

def _look_at_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = target - eye
    z /= max(np.linalg.norm(z), 1e-8)
    x = np.cross(z, np.array([0.0, 0.0, 1.0], np.float32))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([0.0, 1.0, 0.0], np.float32))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def _sample_cameras(bmin: np.ndarray, bmax: np.ndarray, n: int = NUM_VIEWS) -> np.ndarray:
    c, ext = 0.5 * (bmin + bmax), np.maximum(bmax - bmin, 1.0)
    eye_z = bmin[2] + np.clip(0.55 * ext[2], 1.1, 1.8)
    look_z = bmin[2] + np.clip(0.35 * ext[2], 0.6, 1.3)
    poses = []
    for i in range(n):
        th = 2.0 * math.pi * (i / n) + 0.15
        eye = np.array([c[0] + 0.30 * ext[0] * math.cos(th), c[1] + 0.30 * ext[1] * math.sin(th), eye_z + 0.12 * math.sin(2 * th)], np.float32)
        tgt = np.array([c[0] - 0.10 * ext[0] * math.cos(th), c[1] - 0.10 * ext[1] * math.sin(th), look_z], np.float32)
        poses.append(_look_at_c2w(eye, tgt))
    return np.stack(poses)


def download_sage_scenes(num_dev: int, num_val: int, num_test: int, local_sage_dir: str | None = None) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    total = num_dev + num_val + num_test

    if local_sage_dir is not None:
        base = Path(local_sage_dir)
        root = base / "scenes" if (base / "scenes").exists() else base
        cands = sorted(p.name for p in root.iterdir() if p.is_dir())
        assert len(cands) >= total, f"Need {total} scenes in {root}, found {len(cands)}."
        random.Random(42).shuffle(cands)
        selected = cands[:total]
        if not RAW_SAGE_DIR.exists():
            RAW_SAGE_DIR.symlink_to(base.resolve())
    else:
        from huggingface_hub import HfApi, snapshot_download

        print(f"[Stage 0] Listing {SAGE_REPO_ID}/scenes...")
        entries = list(HfApi().list_repo_tree(SAGE_REPO_ID, path_in_repo="scenes", repo_type="dataset"))
        cands = sorted(Path(e.path).name for e in entries if getattr(e, "path", "").startswith("scenes/"))
        assert len(cands) >= total, f"Found {len(cands)} scenes, need {total}."
        stride = max(1, len(cands) // total)
        selected = list(dict.fromkeys(cands[min(i * stride, len(cands) - 1)] for i in range(total)))
        for c in cands:
            if len(selected) >= total:
                break
            if c not in selected:
                selected.append(c)
        print(f"[Stage 0] Downloading {len(selected)} selected scenes -> {RAW_SAGE_DIR}")
        snapshot_download(
            repo_id=SAGE_REPO_ID, repo_type="dataset", local_dir=str(RAW_SAGE_DIR),
            allow_patterns=[f"scenes/{sid}/**" for sid in selected] + ["kits/**"],
        )

    splits = {
        "dev": selected[:num_dev],
        "val": selected[num_dev : num_dev + num_val],
        "test": selected[num_dev + num_val : total],
        "source": "sage-10k",
    }
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))
    print(f"[Stage 0] Splits written: dev={len(splits['dev'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


def _render_sage_scene_bpy(raw_dir: Path, out_dir: Path, cycles_samples: int = 24) -> None:
    import bpy
    from mathutils import Matrix, Quaternion, Vector
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    views_dir = out_dir / "views"
    views_dir.mkdir(exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = cycles_samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        cuda_devs = [d for d in prefs.devices if d.type == "CUDA"]
        for d in cuda_devs:
            d.use = True
        sc.cycles.device = "GPU" if cuda_devs else "CPU"
    except Exception:
        sc.cycles.device = "CPU"

    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = IMG_WIDTH, IMG_HEIGHT, 100
    sc.view_layers[0].use_pass_z = True
    sc.view_layers[0].use_pass_normal = True
    sc.view_layers[0].use_pass_object_index = True

    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.35, 0.35, 0.35, 1.0)
        bg.inputs[1].default_value = 0.6

    inst_meta: dict[int, dict[str, Any]] = {}
    pid = 1

    def _tag_new(existing: set[str], label: str) -> None:
        nonlocal pid
        hit = False
        for o in bpy.data.objects:
            if o.name not in existing and o.type == "MESH":
                o.pass_index = pid
                hit = True
        if hit:
            inst_meta[pid] = {"label": label, "struct_id": classify_label_to_struct(label)}
            pid += 1

    glbs = list(raw_dir.glob("*.glb"))
    if glbs:
        ex = {o.name for o in bpy.data.objects}
        bpy.ops.import_scene.gltf(filepath=str(glbs[0]))
        for o in bpy.data.objects:
            if o.name not in ex and o.type == "MESH":
                o.pass_index = pid
                inst_meta[pid] = {"label": o.name, "struct_id": classify_label_to_struct(o.name)}
                pid += 1
    else:
        jsons = sorted(raw_dir.glob("*.json"))
        layout = json.loads(jsons[0].read_text()) if jsons else {}
        rooms = layout.get("rooms", [layout["room"]] if "room" in layout else [])
        dims = rooms[0].get("dimensions", {}) if rooms and isinstance(rooms[0], dict) else {}
        rw = float(dims.get("width", dims.get("x", 5.0)))
        rl = float(dims.get("length", dims.get("y", 5.0)))
        rh = float(dims.get("height", dims.get("z", 2.8)))

        def _box(name, loc, scale, col, lbl):
            ex = {o.name for o in bpy.data.objects}
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
            o = bpy.context.active_object
            o.name, o.scale = name, scale
            mat = bpy.data.materials.new(f"m_{name}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (*col, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.75
            o.data.materials.append(mat)
            _tag_new(ex, lbl)

        _box("floor", (rw / 2, rl / 2, -0.05), (rw, rl, 0.1), (0.55, 0.45, 0.35), "floor")
        _box("ceiling", (rw / 2, rl / 2, rh + 0.05), (rw, rl, 0.1), (0.88, 0.88, 0.86), "ceiling")
        _box("wall_s", (rw / 2, -0.05, rh / 2), (rw, 0.1, rh), (0.82, 0.80, 0.76), "wall")
        _box("wall_n", (rw / 2, rl + 0.05, rh / 2), (rw, 0.1, rh), (0.80, 0.82, 0.84), "wall")
        _box("wall_w", (-0.05, rl / 2, rh / 2), (0.1, rl, rh), (0.84, 0.82, 0.78), "wall")
        _box("wall_e", (rw + 0.05, rl / 2, rh / 2), (0.1, rl, rh), (0.78, 0.80, 0.82), "wall")

        obj_dir = raw_dir / "objects"
        entries = (rooms[0].get("objects", []) if rooms and isinstance(rooms[0], dict) else []) or layout.get("objects", [])
        imported = set()
        for e in entries:
            if not isinstance(e, dict):
                continue
            lbl = e.get("category", e.get("class", e.get("type", e.get("name", "object"))))
            oid = str(e.get("object_id", e.get("id", e.get("source_id", ""))))
            mp = None
            for key in ("model_path", "mesh_file"):
                if e.get(key):
                    for bd in (raw_dir, obj_dir):
                        c = bd / str(e[key])
                        if c.exists():
                            mp = c
                            break
            if mp is None and obj_dir.exists() and oid:
                for ext in (".glb", ".gltf", ".obj", ".ply"):
                    hits = [obj_dir / f"{oid}{ext}"] if (obj_dir / f"{oid}{ext}").exists() else list(obj_dir.glob(f"*{oid}*{ext}"))
                    if hits:
                        mp = hits[0]
                        break
            if mp is None:
                continue
            imported.add(mp)
            ex = {o.name for o in bpy.data.objects}
            if mp.suffix in (".glb", ".gltf"):
                bpy.ops.import_scene.gltf(filepath=str(mp))
            elif mp.suffix == ".obj":
                bpy.ops.wm.obj_import(filepath=str(mp))
            elif mp.suffix == ".ply":
                bpy.ops.wm.ply_import(filepath=str(mp))

            pos, rot = e.get("position"), e.get("rotation")
            loc = (float(pos["x"]), float(pos["y"]), float(pos["z"])) if isinstance(pos, dict) and {"x", "y", "z"}.issubset(pos) else (tuple(float(v) for v in pos) if isinstance(pos, (list, tuple)) and len(pos) == 3 else None)
            erot, qrot = None, None
            if isinstance(rot, dict):
                erot = tuple(math.radians(float(rot.get(a, 0.0))) for a in ("x", "y", "z"))
            elif isinstance(rot, (list, tuple)) and len(rot) == 3:
                vals = [float(v) for v in rot]
                erot = tuple(math.radians(v) for v in vals) if max(abs(v) for v in vals) > 6.4 else tuple(vals)
            elif isinstance(rot, (list, tuple)) and len(rot) == 4:
                qrot = Quaternion((float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])))

            for o in bpy.data.objects:
                if o.name not in ex and o.parent is None:
                    if loc is not None:
                        o.location = loc
                    if qrot is not None:
                        o.rotation_mode, o.rotation_quaternion = "QUATERNION", qrot
                    elif erot is not None:
                        o.rotation_euler = erot
            _tag_new(ex, lbl)

        if obj_dir.exists() and not imported:
            for mp in sorted(obj_dir.glob("*.glb")):
                ex = {o.name for o in bpy.data.objects}
                bpy.ops.import_scene.gltf(filepath=str(mp))
                _tag_new(ex, mp.stem)

    corners = [list(o.matrix_world @ Vector(c)) for o in bpy.data.objects if o.type == "MESH" for c in o.bound_box]
    arr = np.asarray(corners, np.float32) if corners else np.array([[0, 0, 0], [4, 4, 2.8]], np.float32)
    bmin, bmax = np.percentile(arr, 2, axis=0), np.percentile(arr, 98, axis=0)
    ctr = 0.5 * (bmin + bmax)

    ld = bpy.data.lights.new("Light", type="AREA")
    ld.energy, ld.size = 180.0, 1.5
    lo = bpy.data.objects.new("Light", ld)
    sc.collection.objects.link(lo)
    lo.location = (float(ctr[0]), float(ctr[1]), float(bmax[2] - 0.15))

    cd = bpy.data.cameras.new("Cam")
    cd.lens_unit, cd.angle, cd.clip_start, cd.clip_end = "FOV", math.radians(65.0), 0.05, 50.0
    co = bpy.data.objects.new("Cam", cd)
    sc.collection.objects.link(co)
    sc.camera = co

    fx = 0.5 * IMG_WIDTH / math.tan(cd.angle / 2.0)
    K1 = np.array([[fx, 0, IMG_WIDTH / 2], [0, fx, IMG_HEIGHT / 2], [0, 0, 1]], np.float32)
    gt_K = np.repeat(K1[None], NUM_VIEWS, axis=0)
    gt_poses = _sample_cameras(bmin, bmax, NUM_VIEWS)
    flip_cv_bl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    sc.use_nodes = True
    tree = sc.node_tree
    tree.nodes.clear()
    rl_node = tree.nodes.new("CompositorNodeRLayers")
    tmp_exr = out_dir / "_tmp_exr"
    tmp_exr.mkdir(exist_ok=True)
    out_node = tree.nodes.new("CompositorNodeOutputFile")
    out_node.base_path, out_node.format.file_format, out_node.format.color_depth = str(tmp_exr), "OPEN_EXR", "32"
    out_node.file_slots.clear()
    for s in ("depth_", "normal_", "index_"):
        out_node.file_slots.new(s)
    tree.links.new(rl_node.outputs["Depth"], out_node.inputs["depth_"])
    tree.links.new(rl_node.outputs["Normal"], out_node.inputs["normal_"])
    tree.links.new(rl_node.outputs["IndexOB"], out_node.inputs["index_"])
    sc.render.image_settings.file_format = "PNG"

    def _read_exr(p: Path) -> np.ndarray:
        img = bpy.data.images.load(str(p), check_existing=False)
        buf = np.empty(IMG_HEIGHT * IMG_WIDTH * 4, np.float32)
        img.pixels.foreach_get(buf)
        bpy.data.images.remove(img)
        return buf.reshape(IMG_HEIGHT, IMG_WIDTH, 4)[::-1].copy()

    rgbs, depths, normals, insts = [], [], [], []
    for v in range(NUM_VIEWS):
        sc.frame_set(v)
        co.matrix_world = Matrix((gt_poses[v] @ flip_cv_bl).tolist())
        rp = views_dir / f"frame_{v:03d}.png"
        sc.render.filepath = str(rp)
        bpy.ops.render.render(write_still=True)

        rgbs.append(np.asarray(Image.open(rp).convert("RGB"), np.uint8))
        d = _read_exr(tmp_exr / f"depth_{v:04d}.exr")[..., 0]
        n = _read_exr(tmp_exr / f"normal_{v:04d}.exr")[..., :3]
        idx = np.rint(_read_exr(tmp_exr / f"index_{v:04d}.exr")[..., 0]).astype(np.int32)
        depths.append(np.where((d > 0.05) & (d < TSDF_DEPTH_TRUNC), d, 0.0).astype(np.float32))
        normals.append((n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)).astype(np.float32))
        insts.append(idx)
    shutil.rmtree(tmp_exr, ignore_errors=True)

    rgb, gt_depth, gt_normals, oracle_inst = np.stack(rgbs), np.stack(depths), np.stack(normals), np.stack(insts)
    max_id = max(int(oracle_inst.max()), max(inst_meta.keys(), default=0))
    lut = np.zeros(max_id + 1, dtype=np.uint8)
    for k_id, meta in inst_meta.items():
        lut[k_id] = int(meta["struct_id"])
    oracle_struct = lut[np.clip(oracle_inst, 0, max_id)]
    gt_pts, gt_pnrm = unproject_voxel_surface(gt_depth, gt_poses, gt_K, normals_world=gt_normals)

    np.savez_compressed(
        out_dir / "gt.npz",
        gt_depth=gt_depth, gt_normals=gt_normals, gt_poses=gt_poses, gt_K=gt_K,
        gt_boundary_mask=compute_boundary_mask(gt_depth, oracle_inst),
        gt_points=gt_pts, gt_point_normals=gt_pnrm,
        oracle_instance_id=oracle_inst, oracle_struct_id=oracle_struct,
    )
    np.savez_compressed(out_dir / "rgb.npz", rgb=rgb)


def render_sage_scenes(splits: dict[str, Any], cycles_samples: int = 24) -> None:
    """
    Render all uncached SAGE-10k scenes via headless Blender (bpy==4.3.0).
    If bpy is not installed in the main venv (to avoid NumPy 1.x vs dvlt's NumPy 2.x ABI conflicts),
    automatically runs Stage 1 in a self-contained isolated `uv run --with bpy==4.3.0` subprocess.
    """
    scenes = splits["dev"] + splits["val"] + splits["test"]
    pending = [s for s in scenes if not ((SCENES_DIR / s / "gt.npz").exists() and (SCENES_DIR / s / "rgb.npz").exists())]
    if not pending:
        print(f"[Stage 1] All {len(scenes)} SAGE-10k scenes already rendered.")
        return

    try:
        import bpy  # noqa: F401
        has_bpy = True
    except ImportError:
        has_bpy = False

    if not has_bpy:
        print(f"[Stage 1] Launching isolated uv Blender (bpy==4.3.0) worker for {len(pending)} scenes...")
        cmd = [
            "uv", "run", "--no-project", "--python", "3.11",
            "--extra-index-url", "https://download.blender.org/pypi/",
            "--with", "bpy==4.3.0",
            "--with", "numpy<2",
            "--with", "scipy",
            "--with", "pillow",
            str(Path(__file__).resolve()),
            "--render-only",
            "--cycles-samples", str(cycles_samples),
        ]
        subprocess.run(cmd, check=True)
        return

    root = RAW_SAGE_DIR / "scenes" if (RAW_SAGE_DIR / "scenes").exists() else RAW_SAGE_DIR
    for i, sid in enumerate(scenes, 1):
        out_dir = SCENES_DIR / sid
        if (out_dir / "gt.npz").exists() and (out_dir / "rgb.npz").exists():
            print(f"[Stage 1] ({i}/{len(scenes)}) {sid}: cached.")
            continue
        t0 = time.time()
        print(f"[Stage 1] ({i}/{len(scenes)}) Rendering {sid}...")
        _render_sage_scene_bpy(root / sid, out_dir, cycles_samples=cycles_samples)
        print(f"          rendered in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Stage 1b: Compact Synthetic Benchmark (Zero-Download 30s Smoke Test)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(num_dev: int = 4, num_val: int = 2, num_test: int = 4) -> dict[str, Any]:
    """Generate procedural ray-traced indoor rooms + simulated DVLT & noisy 2D semantics."""
    assert torch is not None, "PyTorch is required."
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SCENES_DIR.mkdir(parents=True, exist_ok=True)

    H, W, N = 140, 182, 8
    total = num_dev + num_val + num_test
    sids = [f"synth_room_{i:03d}" for i in range(total)]
    splits = {"dev": sids[:num_dev], "val": sids[num_dev : num_dev + num_val], "test": sids[num_dev + num_val :], "source": "synthetic"}
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))

    fx = 0.5 * W / math.tan(math.radians(65.0) / 2.0)
    K1 = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], np.float32)
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    d_cam = np.stack([(u - K1[0, 2]) / fx, (v - K1[1, 2]) / fx, np.ones_like(u)], -1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    cos_z = d_cam[..., 2].reshape(-1)
    face_n = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)

    for s_idx, sid in enumerate(sids):
        out_dir = SCENES_DIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1000 + s_idx)
        rx, ry, rz = float(rng.uniform(4.2, 5.4)), float(rng.uniform(4.0, 5.2)), float(rng.uniform(2.6, 3.0))
        cx, cy = rx * 0.5, ry * 0.5
        boxes = [
            (np.array([cx - 0.65, cy - 0.45, 0.68], np.float32), np.array([cx + 0.65, cy + 0.45, 0.76], np.float32), 10, 1, np.array([0.70, 0.52, 0.34], np.float32)),
            (np.array([cx - 0.25, cy - 0.12, 0.76], np.float32), np.array([cx + 0.10, cy + 0.18, 1.15], np.float32), 11, 2, np.array([0.78, 0.79, 0.80], np.float32)),
            (np.array([0.15, cy - 0.60, 0.0], np.float32), np.array([0.65, cy + 0.60, 0.95], np.float32), 12, 1, np.array([0.35, 0.42, 0.52], np.float32)),
            (np.array([cx + 0.85, cy - 0.85, 0.0], np.float32), np.array([cx + 0.95, cy - 0.75, 1.55], np.float32), 13, 3, np.array([0.22, 0.52, 0.28], np.float32)),
        ]
        spheres = [
            (np.array([cx + 0.35, cy - 0.15, 0.94], np.float32), 0.18, 14, 2, np.array([0.78, 0.32, 0.28], np.float32)),
            (np.array([cx + 0.90, cy - 0.80, 1.65], np.float32), 0.28, 13, 3, np.array([0.20, 0.58, 0.26], np.float32)),
        ]
        planes = [
            (2, 0.0, np.array([0, 0, 1], np.float32), 1, np.array([0.62, 0.50, 0.38], np.float32)),
            (2, rz, np.array([0, 0, -1], np.float32), 2, np.array([0.90, 0.90, 0.88], np.float32)),
            (0, 0.0, np.array([1, 0, 0], np.float32), 3, np.array([0.84, 0.82, 0.78], np.float32)),
            (0, rx, np.array([-1, 0, 0], np.float32), 4, np.array([0.78, 0.81, 0.85], np.float32)),
            (1, 0.0, np.array([0, 1, 0], np.float32), 5, np.array([0.82, 0.79, 0.75], np.float32)),
            (1, ry, np.array([0, -1, 0], np.float32), 6, np.array([0.80, 0.83, 0.80], np.float32)),
        ]
        gt_poses = _sample_cameras(np.zeros(3, np.float32), np.array([rx, ry, rz], np.float32), N)
        gt_K = np.repeat(K1[None], N, axis=0)
        rgb_a = np.zeros((N, H, W, 3), np.uint8)
        gt_d = np.zeros((N, H, W), np.float32)
        gt_n = np.zeros((N, H, W, 3), np.float32)
        o_inst = np.zeros((N, H, W), np.int32)
        o_struct = np.zeros((N, H, W), np.uint8)

        for vi in range(N):
            ro = gt_poses[vi, :3, 3]
            rd = (d_cam.reshape(-1, 3) @ gt_poses[vi, :3, :3].T).astype(np.float32)
            inv_d = np.where(np.abs(rd) > 1e-7, 1.0 / np.where(np.abs(rd) > 1e-7, rd, 1.0), 1e7)
            tb = np.full(H * W, 1e6, np.float32)
            nb, cb = np.zeros((H * W, 3), np.float32), np.zeros((H * W, 3), np.float32)
            ib, sb = np.zeros(H * W, np.int32), np.zeros(H * W, np.uint8)
            for ax, val, nrm, pid_p, bcol in planes:
                t = (val - ro[ax]) * inv_d[:, ax]
                hit = (t > 0.05) & (t < tb)
                if not np.any(hit):
                    continue
                p = ro + t[hit, None] * rd[hit]
                ins = (p[:, 0] >= -1e-3) & (p[:, 0] <= rx + 1e-3) & (p[:, 1] >= -1e-3) & (p[:, 1] <= ry + 1e-3) & (p[:, 2] >= -1e-3) & (p[:, 2] <= rz + 1e-3)
                idx = np.where(hit)[0][ins]
                if len(idx) == 0:
                    continue
                pi = p[ins]
                tb[idx], nb[idx], ib[idx], sb[idx] = t[idx], nrm, pid_p, 1
                col = np.broadcast_to(bcol, (len(idx), 3)).copy()
                # Deliberate textured floor/walls to separate RGB edges from true semantic boundaries
                pat = (((np.floor(pi[:, 0] * 4) + np.floor(pi[:, 1] * 4)) % 2) == 0) if pid_p == 1 else (np.sin((pi[:, 1] if ax == 0 else pi[:, 0]) * 14.0) * np.sin(pi[:, 2] * 14.0) > 0.4)
                col[pat] *= 0.84
                cb[idx] = col
            for bmin, bmax, pid_b, st_b, col_b in boxes:
                t0, t1 = (bmin - ro) * inv_d, (bmax - ro) * inv_d
                te, tx = np.max(np.minimum(t0, t1), axis=1), np.min(np.maximum(t0, t1), axis=1)
                idx = np.where((tx >= np.maximum(te, 0.05)) & (te > 0.05) & (te < tb))[0]
                if len(idx):
                    ph = ro + te[idx, None] * rd[idx]
                    face = np.argmin(np.concatenate([np.abs(ph - bmin), np.abs(ph - bmax)], axis=1), axis=1)
                    tb[idx], nb[idx], ib[idx], sb[idx], cb[idx] = te[idx], face_n[face], pid_b, st_b, col_b
            for ctr_s, rad_s, pid_s, st_s, col_s in spheres:
                oc = ro - ctr_s
                b_v = rd @ oc
                disc = b_v * b_v - float(np.dot(oc, oc) - rad_s * rad_s)
                th = -b_v - np.sqrt(np.maximum(disc, 0.0))
                idx = np.where((disc > 0) & (th > 0.05) & (th < tb))[0]
                if len(idx):
                    nh = ro + th[idx, None] * rd[idx] - ctr_s
                    nh /= np.maximum(np.linalg.norm(nh, axis=-1, keepdims=True), 1e-8)
                    tb[idx], nb[idx], ib[idx], sb[idx], cb[idx] = th[idx], nh, pid_s, st_s, col_s
            pts = ro + tb[:, None] * rd
            ld = np.array([cx, cy, rz - 0.15], np.float32) - pts
            ld /= np.maximum(np.linalg.norm(ld, axis=-1, keepdims=True), 1e-8)
            shaded = np.clip(cb * (0.35 + 0.65 * np.clip(np.sum(nb * ld, axis=-1, keepdims=True), 0, 1)), 0, 1)
            rgb_a[vi] = (shaded * 255).astype(np.uint8).reshape(H, W, 3)
            gt_d[vi], gt_n[vi], o_inst[vi], o_struct[vi] = (tb * cos_z).reshape(H, W), nb.reshape(H, W, 3), ib.reshape(H, W), sb.reshape(H, W)

        gt_b = compute_boundary_mask(gt_d, o_inst)
        gt_pts, gt_pn = unproject_voxel_surface(gt_d, gt_poses, gt_K, normals_world=gt_n)

        # Simulated Déjà View predictions (arbitrary scale + HF/LF noise + edge bleeding + pose jitter)
        sc_dvlt = float(rng.uniform(0.7, 1.3))
        pd_d, pd_c, pd_p = np.zeros_like(gt_d), np.zeros_like(gt_d), gt_poses.copy()
        for vi in range(N):
            z = gt_d[vi]
            hf = rng.normal(0, 0.025, z.shape).astype(np.float32)
            lf = ndimage.gaussian_filter(rng.normal(0, 0.14, z.shape), sigma=10.0).astype(np.float32)
            bleed = np.where(gt_b[vi], 0.55 * (ndimage.gaussian_filter(z, 2.2) - z) + rng.normal(0, 0.06, z.shape), 0.0).astype(np.float32)
            pd_d[vi] = np.clip(z + hf + lf + bleed, 0.1, TSDF_DEPTH_TRUNC - 0.1) / sc_dvlt
            pd_c[vi] = np.clip(0.92 - 0.45 * gt_b[vi] - np.abs(hf) * 4.0, 0.1, 1.0)
            pd_p[vi, :3, 3] = (gt_poses[vi, :3, 3] + rng.normal(0, 0.015, 3)) / sc_dvlt
        s0, R0, t0 = compute_fixed_sim3(pd_d, pd_p, gt_K, gt_d, gt_poses, gt_K)

        np.savez_compressed(
            out_dir / "gt.npz",
            gt_depth=gt_d, gt_normals=gt_n, gt_poses=gt_poses, gt_K=gt_K, gt_boundary_mask=gt_b,
            gt_points=gt_pts, gt_point_normals=gt_pn, oracle_instance_id=o_inst, oracle_struct_id=o_struct,
            sim3_scale=np.array(s0), sim3_R=R0, sim3_t=t0,
        )
        torch.save({
            "rgb": torch.from_numpy(rgb_a).permute(0, 3, 1, 2).contiguous(),
            "depth": torch.from_numpy(pd_d), "conf": torch.from_numpy(pd_c.astype(np.float16)),
            "poses": torch.from_numpy(pd_p), "intrinsics": torch.from_numpy(gt_K),
            "dvlt_k": DVLT_DEFAULT_K, "dvlt_runtime_sec": 0.42,
        }, out_dir / f"dvlt_k{DVLT_DEFAULT_K}.pt")

        # Simulated predicted 2D semantics (view-inconsistent IDs, 6% label noise, synthetic DINO)
        sam_i, sam_b = np.zeros_like(o_inst, np.int16), np.zeros_like(gt_d, np.float16)
        sem_s, sem_c = o_struct.copy(), np.full_like(gt_d, 0.85, np.float32)
        hp, wp = H // 14, W // 14
        embs = rng.normal(0, 1, (32, DINO_PCA_DIM)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=-1, keepdims=True)
        dino = np.zeros((N, hp, wp, DINO_PCA_DIM), np.float16)
        for vi in range(N):
            sh = ndimage.shift(o_inst[vi], (rng.uniform(-1, 1), rng.uniform(-1, 1)), order=0, mode="nearest")
            uniq = np.unique(sh)
            perm = rng.permutation(len(uniq)) + 1 + vi * 32
            for u_id, p_id in zip(uniq, perm):
                sam_i[vi][sh == u_id] = int(p_id)
            edge = (sam_i[vi] != np.roll(sam_i[vi], 1, 1)) | (sam_i[vi] != np.roll(sam_i[vi], 1, 0))
            sam_b[vi] = ndimage.gaussian_filter(edge.astype(np.float32), 1.2).astype(np.float16)
            flip = rng.random((H, W)) < 0.06
            sem_s[vi][flip], sem_c[vi][flip] = rng.integers(1, 4, int(flip.sum()), np.uint8), 0.35
            f = embs[np.clip(o_inst[vi, 7::14, 7::14][:hp, :wp], 0, 31)] + rng.normal(0, 0.15, (hp, wp, DINO_PCA_DIM))
            dino[vi] = (f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-6)).astype(np.float16)
        torch.save({
            "sam_instance_id": torch.from_numpy(sam_i), "sam_boundary": torch.from_numpy(sam_b),
            "sem_class_id": torch.from_numpy(sem_s.astype(np.int16)), "sem_class_conf": torch.from_numpy(sem_c.astype(np.float16)),
            "sem_struct_id": torch.from_numpy(sem_s), "dino_feats": torch.from_numpy(dino),
            "sem_extract_sec": {"sam_sec": 0.85, "seg_sec": 0.45, "dino_sec": 0.12, "total_sec": 1.42},
        }, out_dir / "semantics_pred.pt")
        print(f"[Synthetic] {sid} ready")

    compute_and_cache_baselines(splits, ks=(DVLT_DEFAULT_K,))
    return splits


# ---------------------------------------------------------------------------
# Stage 2 & 3: Sequential Déjà View + Semantic Model Caching (RTX 3080-Safe)
# ---------------------------------------------------------------------------

def _parse_dvlt_out(outputs: Any, N: int, hw: tuple[int, int]) -> dict[str, np.ndarray | None]:
    d = outputs if isinstance(outputs, dict) else vars(outputs)

    def _get(keys):
        for k in keys:
            if k in d and d[k] is not None:
                v = d[k][-1] if isinstance(d[k], (list, tuple)) else d[k]
                return v[0] if (v.ndim >= 3 and v.shape[0] == 1 and v.shape[1] == N) else v
        return None

    depth = _get(("depth", "pred_depth", "depths", "depth_map"))
    assert depth is not None, f"Depth missing in DVLT outputs: {list(d.keys())}"
    depth = depth.float().squeeze(-1) if (depth.ndim == 4 and depth.shape[-1] == 1) else (depth.float().squeeze(1) if depth.ndim == 4 else depth.float())
    if depth.shape[-2:] != hw:
        depth = F.interpolate(depth.unsqueeze(1), size=hw, mode="bilinear", align_corners=False).squeeze(1)

    conf = _get(("conf", "confidence", "depth_conf", "pred_conf"))
    if conf is not None:
        conf = conf.float().squeeze(-1) if (conf.ndim == 4 and conf.shape[-1] == 1) else (conf.float().squeeze(1) if conf.ndim == 4 else conf.float())
        if conf.shape[-2:] != hw:
            conf = F.interpolate(conf.unsqueeze(1), size=hw, mode="bilinear", align_corners=False).squeeze(1)
    else:
        conf = torch.ones_like(depth)

    poses = _get(("poses", "extrinsics", "camera_poses", "c2w", "pred_poses"))
    if poses is not None:
        poses = poses.float()
        if poses.shape[-2:] == (3, 4):
            pad = torch.zeros(N, 1, 4, device=poses.device, dtype=poses.dtype)
            pad[:, 0, 3] = 1.0
            poses = torch.cat([poses, pad], dim=1)
        poses_np = poses.detach().cpu().numpy().astype(np.float32)
    else:
        poses_np = np.repeat(np.eye(4, dtype=np.float32)[None], N, axis=0)

    K = _get(("intrinsics", "K", "pred_intrinsics", "camera_intrinsics"))
    return {
        "depth": depth.detach().cpu().numpy().astype(np.float32),
        "conf": conf.detach().cpu().numpy().astype(np.float32),
        "poses": poses_np,
        "intrinsics": K.float().detach().cpu().numpy().astype(np.float32) if K is not None else None,
    }


def cache_dvlt_predictions(splits: dict[str, Any], ks: tuple[int, ...] = (DVLT_DEFAULT_K,)) -> None:
    assert torch is not None and torch.cuda.is_available(), "CUDA required for Déjà View."
    from dvlt.model.dvlt.model import DVLT
    from dvlt.util.preprocess import load_sequence, preprocess_images

    scenes = splits["dev"] + splits["val"] + splits["test"]
    if not any(not (SCENES_DIR / s / f"dvlt_k{k}.pt").exists() for s in scenes for k in ks):
        print(f"[Stage 2] Déjà View K={ks} already cached.")
        return

    print(f"[Stage 2] Loading {DVLT_REPO_ID} on CUDA...")
    dev = torch.device("cuda")
    model = DVLT.from_pretrained(DVLT_REPO_ID).to(dev).eval()
    k_kw = next((n for n in ("K", "num_iters", "num_loops", "n_iters", "steps") if n in inspect.signature(model.forward).parameters), None)

    for sid in scenes:
        sdir = SCENES_DIR / sid
        gt = dict(np.load(sdir / "gt.npz"))
        rgb = np.load(sdir / "rgb.npz")["rgb"]
        N, H, W, _ = rgb.shape
        imgs, _ = load_sequence(str(sdir / "views"))
        batch = preprocess_images(imgs)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()} if isinstance(batch, dict) else batch.to(dev)

        for k in ks:
            cpath = sdir / f"dvlt_k{k}.pt"
            if cpath.exists():
                continue
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch, **({k_kw: k} if k_kw else {}))
            torch.cuda.synchronize()
            dt = time.time() - t0
            parsed = _parse_dvlt_out(out, N, (H, W))
            intr = parsed["intrinsics"] if parsed["intrinsics"] is not None else gt["gt_K"]

            if k == DVLT_DEFAULT_K and "sim3_scale" not in gt:
                s0, R0, t0_v = compute_fixed_sim3(parsed["depth"], parsed["poses"], intr, gt["gt_depth"], gt["gt_poses"], gt["gt_K"])
                gt["sim3_scale"], gt["sim3_R"], gt["sim3_t"] = np.array(s0), R0, t0_v
                np.savez_compressed(sdir / "gt.npz", **gt)

            torch.save({
                "rgb": torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous(),
                "depth": torch.from_numpy(parsed["depth"]),
                "conf": torch.from_numpy(parsed["conf"].astype(np.float16)),
                "poses": torch.from_numpy(parsed["poses"]),
                "intrinsics": torch.from_numpy(intr.astype(np.float32)),
                "dvlt_k": k, "dvlt_runtime_sec": float(dt),
            }, cpath)
            print(f"[Stage 2] {sid} K={k}: {dt:.2f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def cache_predicted_semantics(splits: dict[str, Any]) -> None:
    assert torch is not None and torch.cuda.is_available(), "CUDA required for semantic extraction."
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoModelForUniversalSegmentation, pipeline

    scenes = splits["dev"] + splits["val"] + splits["test"]
    todo = [s for s in scenes if not (SCENES_DIR / s / "semantics_pred.pt").exists()]
    if not todo:
        print("[Stage 3] Predicted semantics already cached.")
        return

    dev = torch.device("cuda")
    buf: dict[str, dict[str, Any]] = {s: {"sem_extract_sec": {}} for s in todo}

    # 3A: SAM 2
    print(f"[Stage 3A] Loading {SAM2_MODEL_ID}...")
    sam = pipeline("mask-generation", model=SAM2_MODEL_ID, device=0, torch_dtype=torch.float16)
    for sid in todo:
        rgb = np.load(SCENES_DIR / sid / "rgb.npz")["rgb"]
        N, H, W, _ = rgb.shape
        s_inst, s_bnd = np.zeros((N, H, W), np.int16), np.zeros((N, H, W), np.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        for vi in range(N):
            out = sam(Image.fromarray(rgb[vi]), points_per_batch=64)
            masks, scores = out.get("masks", []), out.get("scores", [])
            order = sorted(range(len(masks)), key=lambda i: int(np.sum(masks[i])), reverse=True)
            imap, bacc = np.zeros((H, W), np.int16), np.zeros((H, W), np.float32)
            for nid, mi in enumerate(order, 1):
                m = np.asarray(masks[mi], bool)
                if m.shape != (H, W) or m.sum() < 16:
                    continue
                imap[m] = min(nid, 32767)
                sc_m = float(scores[mi]) if mi < len(scores) else 1.0
                edge = (m != np.roll(m, 1, 1)) | (m != np.roll(m, 1, 0))
                bacc = np.maximum(bacc, edge.astype(np.float32) * sc_m)
            s_inst[vi], s_bnd[vi] = imap, ndimage.gaussian_filter(bacc, 1.0).astype(np.float16)
        torch.cuda.synchronize()
        buf[sid]["sam_instance_id"], buf[sid]["sam_boundary"] = torch.from_numpy(s_inst), torch.from_numpy(s_bnd)
        buf[sid]["sem_extract_sec"]["sam_sec"] = float(time.time() - t0)
    del sam
    gc.collect()
    torch.cuda.empty_cache()

    # 3B: Mask2Former ADE20K
    print(f"[Stage 3B] Loading {SEG_MODEL_ID}...")
    sproc = AutoImageProcessor.from_pretrained(SEG_MODEL_ID)
    smod = AutoModelForUniversalSegmentation.from_pretrained(SEG_MODEL_ID).to(dev).eval()
    max_c = max(int(k) for k in smod.config.id2label)
    c2s = np.zeros(max_c + 1, np.uint8)
    for k_id, lname in smod.config.id2label.items():
        c2s[int(k_id)] = classify_label_to_struct(lname)
    for sid in todo:
        rgb = np.load(SCENES_DIR / sid / "rgb.npz")["rgb"]
        N, H, W, _ = rgb.shape
        cids, cconf, sids_arr = np.zeros((N, H, W), np.int16), np.zeros((N, H, W), np.float16), np.zeros((N, H, W), np.uint8)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            for vi in range(N):
                inp = sproc(images=Image.fromarray(rgb[vi]), return_tensors="pt").to(dev)
                out = smod(**inp)
                cid = sproc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].cpu().numpy().astype(np.int16)
                cids[vi], sids_arr[vi] = cid, c2s[np.clip(cid, 0, max_c)]
                cp = out.class_queries_logits[0].softmax(-1)[:, :-1]
                mp = out.masks_queries_logits[0].sigmoid()
                sp = F.interpolate(torch.einsum("qc,qhw->chw", cp, mp).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
                cconf[vi] = sp.max(0).values.clamp(0, 1).cpu().numpy().astype(np.float16)
        torch.cuda.synchronize()
        buf[sid]["sem_class_id"], buf[sid]["sem_class_conf"], buf[sid]["sem_struct_id"] = torch.from_numpy(cids), torch.from_numpy(cconf), torch.from_numpy(sids_arr)
        buf[sid]["sem_extract_sec"]["seg_sec"] = float(time.time() - t0)
    del smod, sproc
    gc.collect()
    torch.cuda.empty_cache()

    # 3C: DINOv2 (PCA-64 fp16)
    print(f"[Stage 3C] Loading {DINO_MODEL_ID}...")
    dproc = AutoImageProcessor.from_pretrained(DINO_MODEL_ID, do_resize=False, do_center_crop=False)
    dmod = AutoModel.from_pretrained(DINO_MODEL_ID).to(dev).eval()
    for sid in todo:
        rgb = np.load(SCENES_DIR / sid / "rgb.npz")["rgb"]
        N, H, W, _ = rgb.shape
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            inp = dproc(images=[Image.fromarray(rgb[vi]) for vi in range(N)], return_tensors="pt").to(dev)
            ft = F.normalize(dmod(**inp).last_hidden_state[:, 1:, :].float(), dim=-1)
            flat = ft.reshape(-1, ft.shape[-1])
            fc = flat - flat.mean(0, keepdim=True)
            _, _, V = torch.pca_lowrank(fc, q=min(DINO_PCA_DIM, flat.shape[-1]))
            proj = F.normalize(fc @ V[:, :DINO_PCA_DIM], dim=-1).reshape(N, H // 14, W // 14, -1).half().cpu()
        torch.cuda.synchronize()
        buf[sid]["dino_feats"] = proj
        sec = buf[sid]["sem_extract_sec"]
        sec["dino_sec"] = float(time.time() - t0)
        sec["total_sec"] = sec["sam_sec"] + sec["seg_sec"] + sec["dino_sec"]
        torch.save(buf[sid], SCENES_DIR / sid / "semantics_pred.pt")
        print(f"[Stage 3] {sid}: total semantic extraction {sec['total_sec']:.2f}s")
    del dmod, dproc
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Runtime Utilities & Anti-Cheat Evaluation Harness (Imported by train.py)
# ---------------------------------------------------------------------------

def get_split_scenes(split: str = "dev", allow_test: bool = False) -> list[str]:
    assert SPLITS_PATH.exists(), f"Missing {SPLITS_PATH}. Run `uv run prepare.py` first."
    if split == "test" and not allow_test:
        raise PermissionError("Locked blind 'test' split cannot be accessed during autonomous experimentation.")
    splits = json.loads(SPLITS_PATH.read_text())
    return list(splits[split])


def load_scene(scene_id: str, mode: str = "none", dvlt_k: int = DVLT_DEFAULT_K, device: str = "cuda") -> dict[str, Any]:
    """Load cached scene inputs with strict mode isolation (none / oracle / predicted). Never exposes GT geometry."""
    assert mode in ("none", "oracle", "predicted"), f"Invalid mode: {mode}"
    sdir = SCENES_DIR / scene_id
    raw = torch.load(sdir / f"dvlt_k{dvlt_k}.pt", map_location="cpu", weights_only=False)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    scene: dict[str, Any] = {
        "scene_id": scene_id, "mode": mode, "dvlt_k": dvlt_k,
        "rgb": raw["rgb"].to(dev).float() / 255.0,
        "depth": raw["depth"].to(dev).float(),
        "conf": raw["conf"].to(dev).float(),
        "poses": raw["poses"].to(dev).float(),
        "intrinsics": raw["intrinsics"].to(dev).float(),
        "dvlt_runtime_sec": float(raw.get("dvlt_runtime_sec", 0.0)),
        "instance_id": None, "struct_id": None, "boundary_prob": None,
        "class_id": None, "class_conf": None, "dino_feats": None, "sem_extract_sec": 0.0,
    }
    if mode == "oracle":
        gt = np.load(sdir / "gt.npz")
        inst, struct = gt["oracle_instance_id"], gt["oracle_struct_id"]
        bprob = np.zeros(inst.shape, np.float32)
        for vi in range(inst.shape[0]):
            edge = (inst[vi] != np.roll(inst[vi], 1, 1)) | (inst[vi] != np.roll(inst[vi], 1, 0))
            edge[:, 0] = False
            edge[0, :] = False
            bprob[vi] = ndimage.gaussian_filter(edge.astype(np.float32), 0.8)
        scene["instance_id"] = torch.from_numpy(inst.astype(np.int32)).to(dev)
        scene["struct_id"] = torch.from_numpy(struct.astype(np.int64)).to(dev)
        scene["boundary_prob"] = torch.from_numpy(bprob).to(dev)
        scene["class_conf"] = torch.ones(inst.shape, dtype=torch.float32, device=dev)
    elif mode == "predicted":
        sem = torch.load(sdir / "semantics_pred.pt", map_location="cpu", weights_only=False)
        scene["instance_id"] = sem["sam_instance_id"].to(dev).long()
        scene["struct_id"] = sem["sem_struct_id"].to(dev).long()
        scene["boundary_prob"] = sem["sam_boundary"].to(dev).float()
        scene["class_id"] = sem["sem_class_id"].to(dev).long()
        scene["class_conf"] = sem["sem_class_conf"].to(dev).float()
        scene["dino_feats"] = sem["dino_feats"].to(dev).float()
        sec = sem.get("sem_extract_sec", {})
        scene["sem_extract_sec"] = float(sec.get("total_sec", 0.0)) if isinstance(sec, dict) else float(sec)
    return scene


def evaluate_scene(scene_id: str, pred_depth: Any, pred_poses: Any, pred_valid_mask: Any = None, dvlt_k: int = DVLT_DEFAULT_K) -> dict[str, float]:
    """Anti-cheat single-scene GT evaluation using frozen raw-DVLT Sim(3) + fixed TSDF surface."""
    sdir = SCENES_DIR / scene_id
    gt = np.load(sdir / "gt.npz")
    raw = torch.load(sdir / f"dvlt_k{dvlt_k}.pt", map_location="cpu", weights_only=False)

    to_np = lambda x: x.detach().cpu().numpy() if (torch is not None and torch.is_tensor(x)) else np.asarray(x)
    pd_d = to_np(pred_depth).astype(np.float32)
    pd_p = to_np(pred_poses).astype(np.float32)
    pd_m = to_np(pred_valid_mask).astype(bool) if pred_valid_mask is not None else None

    if not (np.all(np.isfinite(pd_d)) and np.all(np.isfinite(pd_p))):
        return {"acc_cm": 999.0, "comp_cm": 999.0, "cd_cm": 999.0, "f1_2cm": 0.0, "f1_5cm": 0.0, "nae_deg": 90.0, "edge_err_cm": 999.0, "coverage": 0.0}

    d_al, p_al = apply_sim3(pd_d, pd_p, float(gt["sim3_scale"]), gt["sim3_R"], gt["sim3_t"])
    gt_d, gt_b = gt["gt_depth"], gt["gt_boundary_mask"].astype(bool)
    gt_v = (gt_d > 0.05) & (gt_d < TSDF_DEPTH_TRUNC)
    pr_v = (d_al > 0.05) & (d_al < TSDF_DEPTH_TRUNC) & (pd_m if pd_m is not None else True)
    coverage = float(np.sum(pr_v & gt_v) / max(np.sum(gt_v), 1))

    pred_pts, pred_nrm = extract_surface_tsdf(d_al, p_al, raw["intrinsics"].numpy(), valid_masks=pr_v)
    if len(pred_pts) < 100:
        return {"acc_cm": 999.0, "comp_cm": 999.0, "cd_cm": 999.0, "f1_2cm": 0.0, "f1_5cm": 0.0, "nae_deg": 90.0, "edge_err_cm": 999.0, "coverage": coverage}

    gt_pts, gt_nrm = gt["gt_points"], gt["gt_point_normals"]
    d_p2g, _ = cKDTree(gt_pts).query(pred_pts, k=1)
    d_g2p, idx_g2p = cKDTree(pred_pts).query(gt_pts, k=1)

    acc_cm, comp_cm = float(np.mean(d_p2g) * 100.0), float(np.mean(d_g2p) * 100.0)
    f1 = lambda th: 2.0 * float(np.mean(d_p2g < th)) * float(np.mean(d_g2p < th)) / max(float(np.mean(d_p2g < th)) + float(np.mean(d_g2p < th)), 1e-8)

    cos_s = np.abs(np.sum(gt_nrm * pred_nrm[idx_g2p], axis=-1))
    ang = np.where(d_g2p <= NORMAL_MATCH_RADIUS, np.degrees(np.arccos(np.clip(cos_s, 0.0, 1.0))), 90.0)
    bmask = gt_b & gt_v
    edge_err_cm = float(np.mean(np.minimum(np.where(pr_v[bmask], np.abs(d_al[bmask] - gt_d[bmask]), 0.50), 1.0)) * 100.0) if np.any(bmask) else 0.5 * (acc_cm + comp_cm)

    return {
        "acc_cm": acc_cm, "comp_cm": comp_cm, "cd_cm": 0.5 * (acc_cm + comp_cm),
        "f1_2cm": f1(FSCORE_THRESH), "f1_5cm": f1(FSCORE_THRESH_LOOSE),
        "nae_deg": float(np.mean(ang)), "edge_err_cm": edge_err_cm, "coverage": coverage,
    }


def _baseline_path(split: str, dvlt_k: int = DVLT_DEFAULT_K) -> Path:
    return CACHE_DIR / f"baseline_{split}_k{dvlt_k}.json"


def compute_and_cache_baselines(splits: dict[str, Any], ks: tuple[int, ...] = (DVLT_DEFAULT_K,)) -> None:
    for k in ks:
        for sp in ("dev", "val", "test"):
            per_scene, rts = {}, []
            for sid in splits.get(sp, []):
                p = SCENES_DIR / sid / f"dvlt_k{k}.pt"
                if not p.exists():
                    continue
                raw = torch.load(p, map_location="cpu", weights_only=False)
                per_scene[sid] = evaluate_scene(sid, raw["depth"].numpy(), raw["poses"].numpy(), dvlt_k=k)
                rts.append(float(raw.get("dvlt_runtime_sec", 0.0)))
            if not per_scene:
                continue
            agg = {m: float(np.mean([v[m] for v in per_scene.values()])) for m in next(iter(per_scene.values()))}
            agg["dvlt_runtime_sec"] = float(np.mean(rts))
            _baseline_path(sp, k).write_text(json.dumps({"split": sp, "dvlt_k": k, "aggregate": agg, "per_scene": per_scene}, indent=2))
            print(f"[Baseline] {sp:4s} K={k:2d} | CD={agg['cd_cm']:.2f}cm | F1@2cm={agg['f1_2cm']:.3f} | NAE={agg['nae_deg']:.1f}deg | Edge={agg['edge_err_cm']:.2f}cm")


def evaluate_reconstruction(predictions: dict[str, dict[str, Any]], split: str = "dev", dvlt_k: int = DVLT_DEFAULT_K, allow_test: bool = False) -> dict[str, Any]:
    """Fixed split-level evaluation returning baseline-normalized val_score (raw DVLT = 1.000000)."""
    sids = get_split_scenes(split, allow_test=allow_test)
    base = json.loads(_baseline_path(split, dvlt_k).read_text())["aggregate"]
    per_scene = {
        sid: evaluate_scene(sid, predictions[sid]["depth"], predictions[sid]["poses"], predictions[sid].get("valid_mask"), dvlt_k=dvlt_k)
        for sid in sids
    }
    agg = {m: float(np.mean([v[m] for v in per_scene.values()])) for m in next(iter(per_scene.values()))}
    val_score = (
        SCORE_WEIGHTS["cd"] * (agg["cd_cm"] / max(base["cd_cm"], 1e-6))
        + SCORE_WEIGHTS["f1_err"] * ((1.0 - agg["f1_2cm"]) / max(1.0 - base["f1_2cm"], 1e-6))
        + SCORE_WEIGHTS["nae"] * (agg["nae_deg"] / max(base["nae_deg"], 1e-6))
        + SCORE_WEIGHTS["edge"] * (agg["edge_err_cm"] / max(base["edge_err_cm"], 1e-6))
    )
    if agg["coverage"] < MIN_COVERAGE_RATIO:
        val_score += 2.0 * (MIN_COVERAGE_RATIO - agg["coverage"])
    agg["val_score"] = float(val_score)
    agg["per_scene"] = per_scene
    return agg


def save_visual_audit(
    predictions: dict[str, dict[str, Any]],
    split: str = "dev",
    dvlt_k: int = DVLT_DEFAULT_K,
    out_dir: str = "visuals",
    max_scenes: int = 2,
    allow_test: bool = False,
) -> None:
    """Save side-by-side RGB, GT/raw/refined normal maps & depth error heatmaps for visual auditing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for sid in get_split_scenes(split, allow_test=allow_test)[:max_scenes]:
        sdir = SCENES_DIR / sid
        gt = np.load(sdir / "gt.npz")
        raw = torch.load(sdir / f"dvlt_k{dvlt_k}.pt", map_location="cpu", weights_only=False)
        pred = predictions[sid]
        rgb = raw["rgb"].permute(0, 2, 3, 1).numpy()
        K = raw["intrinsics"].numpy()
        s0, R0, t0 = float(gt["sim3_scale"]), gt["sim3_R"], gt["sim3_t"]
        raw_d, _ = apply_sim3(raw["depth"].numpy(), raw["poses"].numpy(), s0, R0, t0)
        ref_d, _ = apply_sim3(pred["depth"], pred["poses"], s0, R0, t0)
        gt_d, gt_n, gt_p = gt["gt_depth"], gt["gt_normals"], gt["gt_poses"]

        views = [v for v in (0, 3, 6) if v < rgb.shape[0]]
        fig, axes = plt.subplots(len(views), 6, figsize=(19, 3 * len(views)))
        if len(views) == 1:
            axes = axes[None, :]
        titles = ["RGB", "GT Normals", "Raw DVLT Normals", "Refined Normals", "Raw |Δz| cm", "Refined |Δz| cm"]
        for r, v in enumerate(views):
            panels = [
                rgb[v],
                np.clip(0.5 * (gt_n[v] @ gt_p[v, :3, :3] + 1.0), 0, 1),
                np.clip(0.5 * (compute_depth_normals_np(raw_d[v], K[v]) + 1.0), 0, 1),
                np.clip(0.5 * (compute_depth_normals_np(ref_d[v], K[v]) + 1.0), 0, 1),
                np.abs(raw_d[v] - gt_d[v]) * 100.0,
                np.abs(ref_d[v] - gt_d[v]) * 100.0,
            ]
            for c, p in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(p, cmap="inferno", vmin=0, vmax=15) if c >= 4 else ax.imshow(p)
                if r == 0:
                    ax.set_title(titles[c], fontsize=10)
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path / f"{sid}_audit.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare cached SAGE-10k + Déjà View + Semantics benchmark")
    parser.add_argument("--synthetic", action="store_true", help="Generate fast synthetic benchmark (zero download)")
    parser.add_argument("--local-sage-dir", type=str, default=None, help="Existing local SAGE-10k path")
    parser.add_argument("--num-dev", type=int, default=SPLIT_SIZES["dev"])
    parser.add_argument("--num-val", type=int, default=SPLIT_SIZES["val"])
    parser.add_argument("--num-test", type=int, default=SPLIT_SIZES["test"])
    parser.add_argument("--pareto-ks", action="store_true", help="Cache DVLT for K in {4, 8, 12, 16}")
    parser.add_argument("--cycles-samples", type=int, default=24)
    parser.add_argument("--render-only", action="store_true", help="Only run Stage 1 headless Blender rendering")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    ks = DVLT_PARETO_KS if args.pareto_ks else (DVLT_DEFAULT_K,)
    if args.synthetic:
        generate_synthetic_dataset(args.num_dev, args.num_val, args.num_test)
    elif args.render_only:
        render_sage_scenes(json.loads(SPLITS_PATH.read_text()), cycles_samples=args.cycles_samples)
    elif args.baselines_only:
        compute_and_cache_baselines(json.loads(SPLITS_PATH.read_text()), ks=ks)
    else:
        splits = download_sage_scenes(args.num_dev, args.num_val, args.num_test, args.local_sage_dir)
        render_sage_scenes(splits, cycles_samples=args.cycles_samples)
        cache_dvlt_predictions(splits, ks=ks)
        cache_predicted_semantics(splits)
        compute_and_cache_baselines(splits, ks=ks)
    print("Done! Ready to run `uv run train.py`.")
