"""
Autoresearch post-hoc 3D refinement script. Single-GPU, single-file.
THIS IS THE ONLY FILE THE AGENT EDITS during autonomous experimentation.

Usage:
    uv run train.py                         # Uses MODE / SPLIT / DVLT_K below
    uv run train.py --mode none             # Track A: non-semantic control
    uv run train.py --mode oracle           # Track B: oracle SAGE-10k semantics (upper bound)
    uv run train.py --mode predicted        # Track C: SAM 2 + Mask2Former + DINOv2
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    DVLT_DEFAULT_K,
    TIME_BUDGET,
    evaluate_reconstruction,
    get_split_scenes,
    load_scene,
    save_visual_audit,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit directly — no CLI flags needed for experiments)
# ---------------------------------------------------------------------------

# Experiment track (set per branch: "none" -> "oracle" -> "predicted")
MODE = "none"
SPLIT = "dev"
DVLT_K = DVLT_DEFAULT_K

# Optimization schedule (NUM_STEPS = 0 is the exact raw Déjà View baseline)
NUM_STEPS = 0
LR_DEPTH = 0.015
LR_POSE = 0.0                  # Set > 0 (e.g. 5e-4) to enable joint SE(3) pose refinement
WARMUP_FRAC = 0.05
FINAL_LR_FRAC = 0.10

# Geometry & photometric loss weights (Track A: active in all modes)
W_DATA = 1.0                   # Confidence-weighted fidelity to raw Déjà View depth
W_MULTIVIEW = 0.5              # Cross-view depth & normal reprojection consistency
W_CURVATURE = 0.15             # 2nd-order depth curvature smoothness (preserves tilted planes)
W_NORMAL_SMOOTH = 0.15         # Edge-aware surface normal smoothness
RGB_EDGE_SIGMA = 0.08          # RGB gradient edge-stopping scale
HUBER_DELTA = 0.05             # Robust Huber transition (raw depth units)

# Semantic-guided weights (Track B & C: automatically inactive when MODE == "none")
W_SEM_BOUNDARY = 1.0           # Suppress smoothing across semantic / instance boundaries
W_PLANAR = 0.0                 # Soft per-instance planarity prior for struct_id == 1 (planar_stuff)
W_RIGID_DETAIL = 0.0           # Extra detail fidelity for struct_id == 2 (rigid_thing)
W_THIN_PROTECT = 0.0           # Anti-oversmoothing protection for struct_id == 3 (thin_nonrigid)
W_DINO_CONSISTENCY = 0.0       # Cross-view DINOv2 feature reprojection consistency


# ---------------------------------------------------------------------------
# Differentiable 3D Geometry & SE(3) Primitives
# ---------------------------------------------------------------------------

def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map from axis-angle (M, 3) -> rotation matrices (M, 3, 3)."""
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = omega / theta
    kx, ky, kz = k.unbind(dim=-1)
    zeros = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zeros, -kz, ky], dim=-1),
        torch.stack([kz, zeros, -kx], dim=-1),
        torch.stack([-ky, kx, zeros], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    return eye + sin_t * K + (1.0 - cos_t) * (K @ K)


def unproject_depth(depth: torch.Tensor, K: torch.Tensor, poses_c2w: torch.Tensor | None = None) -> torch.Tensor:
    """Back-project depth (N, H, W) into camera-space or world-space 3D points (N, H, W, 3)."""
    N, H, W = depth.shape
    v, u = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = K[:, 0, 0].view(N, 1, 1).clamp_min(1e-6)
    fy = K[:, 1, 1].view(N, 1, 1).clamp_min(1e-6)
    cx = K[:, 0, 2].view(N, 1, 1)
    cy = K[:, 1, 2].view(N, 1, 1)
    pts_c = torch.stack([(u.unsqueeze(0) - cx) * depth / fx, (v.unsqueeze(0) - cy) * depth / fy, depth], dim=-1)
    if poses_c2w is None:
        return pts_c
    return torch.einsum("nij,nhwj->nhwi", poses_c2w[:, :3, :3], pts_c) + poses_c2w[:, None, None, :3, 3]


def compute_depth_normals(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Differentiable camera-space unit surface normals (N, H, W, 3) oriented toward camera (-Z)."""
    pts = unproject_depth(depth, K, poses_c2w=None)
    dx = F.pad((pts[:, :, 2:, :] - pts[:, :, :-2, :]).permute(0, 3, 1, 2), (1, 1, 0, 0), mode="replicate").permute(0, 2, 3, 1)
    dy = F.pad((pts[:, 2:, :, :] - pts[:, :-2, :, :]).permute(0, 3, 1, 2), (0, 0, 1, 1), mode="replicate").permute(0, 2, 3, 1)
    n = F.normalize(torch.cross(dy, dx, dim=-1), dim=-1, eps=1e-6)
    return n * torch.where(n[..., 2:3] > 0, -1.0, 1.0)


def rgb_edge_weights(rgb: torch.Tensor, sigma: float = RGB_EDGE_SIGMA) -> tuple[torch.Tensor, torch.Tensor]:
    """Anisotropic horizontal (N, H, W-1) and vertical (N, H-1, W) edge-stopping weights from RGB."""
    gx = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs().mean(dim=1)
    gy = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=1)
    return torch.exp(-gx / max(sigma, 1e-4)), torch.exp(-gy / max(sigma, 1e-4))


def semantic_edge_weights(
    instance_id: torch.Tensor | None,
    boundary_prob: torch.Tensor | None,
    rgb_wx: torch.Tensor,
    rgb_wy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Modulate edge-stopping weights with instance masks & soft boundaries when available."""
    if instance_id is None and boundary_prob is None:
        return rgb_wx, rgb_wy
    wx, wy = rgb_wx, rgb_wy
    if instance_id is not None:
        wx = (instance_id[:, :, 1:] == instance_id[:, :, :-1]).float()
        wy = (instance_id[:, 1:, :] == instance_id[:, :-1, :]).float()
    if boundary_prob is not None:
        wx = wx * (1.0 - torch.maximum(boundary_prob[:, :, 1:], boundary_prob[:, :, :-1]).clamp(0.0, 1.0))
        wy = wy * (1.0 - torch.maximum(boundary_prob[:, 1:, :], boundary_prob[:, :-1, :]).clamp(0.0, 1.0))
    return wx, wy


# ---------------------------------------------------------------------------
# Differentiable Refinement Losses
# ---------------------------------------------------------------------------

def multiview_reprojection_loss(
    depth: torch.Tensor,
    poses_c2w: torch.Tensor,
    K: torch.Tensor,
    conf: torch.Tensor,
    offsets: tuple[int, ...] = (1, -1),
    delta: float = HUBER_DELTA,
) -> torch.Tensor:
    """Vectorized cross-view depth reprojection consistency with soft occlusion gating."""
    N, H, W = depth.shape
    pts_w = unproject_depth(depth, K, poses_c2w)
    w2c = torch.linalg.inv(poses_c2w)

    total = depth.new_tensor(0.0)
    used = 0
    for off in offsets:
        j = (torch.arange(N, device=depth.device) + off) % N
        pts_j = torch.einsum("nij,nhwj->nhwi", w2c[j, :3, :3], pts_w) + w2c[j, None, None, :3, 3]
        z_p = pts_j[..., 2].clamp_min(1e-4)
        K_j = K[j]
        u_j = pts_j[..., 0] * K_j[:, 0, 0].view(N, 1, 1) / z_p + K_j[:, 0, 2].view(N, 1, 1)
        v_j = pts_j[..., 1] * K_j[:, 1, 1].view(N, 1, 1) / z_p + K_j[:, 1, 2].view(N, 1, 1)

        gx = 2.0 * (u_j / max(W - 1, 1)) - 1.0
        gy = 2.0 * (v_j / max(H - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        in_fov = (pts_j[..., 2] > 0.05) & (gx >= -1.0) & (gx <= 1.0) & (gy >= -1.0) & (gy <= 1.0)

        z_s = F.grid_sample(depth[j].unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(1)
        occ_ok = in_fov & ((z_p - z_s).abs() / z_p.detach().clamp_min(0.1) < 0.20)
        if occ_ok.any():
            w = conf * occ_ok.float()
            loss_ij = F.huber_loss(z_p, z_s.detach(), reduction="none", delta=delta)
            total = total + (loss_ij * w).sum() / w.sum().clamp_min(1.0)
            used += 1
    return total / max(used, 1)


def curvature_smoothness_loss(depth: torch.Tensor, wx: torch.Tensor, wy: torch.Tensor) -> torch.Tensor:
    """2nd-order (Laplacian) depth smoothness — preserves slanted planes without 1st-order TV staircasing."""
    d2x = (depth[:, :, 2:] - 2.0 * depth[:, :, 1:-1] + depth[:, :, :-2]).abs()
    d2y = (depth[:, 2:, :] - 2.0 * depth[:, 1:-1, :] + depth[:, :-2, :]).abs()
    wx2 = torch.minimum(wx[:, :, 1:], wx[:, :, :-1])
    wy2 = torch.minimum(wy[:, 1:, :], wy[:, :-1, :])
    return (wx2 * d2x).mean() + (wy2 * d2y).mean()


def planar_prior_loss(
    pts_w: torch.Tensor,
    struct_id: torch.Tensor | None,
    instance_id: torch.Tensor | None,
    conf: torch.Tensor,
    min_pixels: int = 256,
) -> torch.Tensor:
    """
    Soft per-instance planarity prior for planar_stuff (struct_id == 1) instances:
    penalizes the smallest eigenvalue of each planar region's 3D point covariance.
    """
    if struct_id is None or instance_id is None:
        return pts_w.new_tensor(0.0)

    planar_mask = struct_id == 1
    if not planar_mask.any():
        return pts_w.new_tensor(0.0)

    # Fit soft planes per (view, instance) on planar pixels
    N = pts_w.shape[0]
    losses = []
    for vi in range(N):
        pm = planar_mask[vi]
        if pm.sum() < min_pixels:
            continue
        inst_v = instance_id[vi][pm]
        pts_v = pts_w[vi][pm]
        w_v = conf[vi][pm].clamp(0.05, 1.0)
        for uid in torch.unique(inst_v):
            if uid <= 0:
                continue
            m = inst_v == uid
            if m.sum() < min_pixels:
                continue
            p = pts_v[m]
            w = w_v[m]
            w = w / w.sum().clamp_min(1e-6)
            mu = (p * w[:, None]).sum(dim=0, keepdim=True)
            pc = p - mu
            cov = (pc * w[:, None]).T @ pc
            eigvals = torch.linalg.eigvalsh(cov)  # ascending: smallest eigenvalue = plane thickness^2
            losses.append(eigvals[0].clamp_min(0.0).sqrt())

    if not losses:
        return pts_w.new_tensor(0.0)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Refinement Model & Milestone 2 Cross-View Semantic Lifting
# ---------------------------------------------------------------------------

class RefinementModel(nn.Module):
    """Differentiable per-scene post-hoc refinement parameterization (depth + anchored SE(3) poses)."""

    def __init__(self, init_depth: torch.Tensor, init_poses: torch.Tensor, refine_poses: bool = False):
        super().__init__()
        self.register_buffer("log_d0", init_depth.clamp_min(1e-3).log())
        self.register_buffer("init_poses", init_poses)
        self.delta_log_depth = nn.Parameter(torch.zeros_like(init_depth))
        self.refine_poses = refine_poses
        N = init_poses.shape[0]
        # Anchor view 0 to identity so the global coordinate frame stays aligned to frozen Sim(3)
        self.delta_rot = nn.Parameter(torch.zeros(max(N - 1, 1), 3, device=init_poses.device), requires_grad=refine_poses)
        self.delta_trans = nn.Parameter(torch.zeros(max(N - 1, 1), 3, device=init_poses.device), requires_grad=refine_poses)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = (self.log_d0 + self.delta_log_depth).exp()
        if not self.refine_poses or self.init_poses.shape[0] <= 1:
            poses = self.init_poses
        else:
            dR = so3_exp_map(self.delta_rot)
            p_list = [self.init_poses[0:1]]
            R_upd = dR @ self.init_poses[1:, :3, :3]
            t_upd = self.init_poses[1:, :3, 3] + self.delta_trans
            p_rest = self.init_poses[1:].clone()
            p_rest[:, :3, :3] = R_upd
            p_rest[:, :3, 3] = t_upd
            p_list.append(p_rest)
            poses = torch.cat(p_list, dim=0)
        valid_mask = torch.ones_like(depth, dtype=torch.bool)
        return depth, poses, valid_mask


def lift_semantics_3d(scene: dict[str, Any]) -> dict[str, Any]:
    """
    Milestone 2 hook: associate noisy 2D semantic cues (SAM 2 masks, Mask2Former labels,
    DINOv2 features) across views into 3D-consistent representations.
    Replace/extend this function during Track C!
    """
    return {
        "instance_id": scene["instance_id"],
        "struct_id": scene["struct_id"],
        "boundary_prob": scene["boundary_prob"],
        "class_conf": scene["class_conf"],
        "dino_feats": scene["dino_feats"],
    }


# ---------------------------------------------------------------------------
# Optimization Loop (Fixed Per-Scene Time Budget)
# ---------------------------------------------------------------------------

def get_lr_mult(progress: float) -> float:
    if WARMUP_FRAC > 0 and progress < WARMUP_FRAC:
        return progress / WARMUP_FRAC
    decay_prog = (progress - WARMUP_FRAC) / max(1.0 - WARMUP_FRAC, 1e-6)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(decay_prog, 0.0), 1.0)))
    return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * cosine


def refine_scene(scene: dict[str, Any], time_budget_sec: float) -> dict[str, torch.Tensor]:
    """Optimize RefinementModel for a single scene within time_budget_sec."""
    init_depth = scene["depth"]
    init_poses = scene["poses"]
    init_conf = scene["conf"].clamp(0.05, 1.0)
    rgb = scene["rgb"]
    K = scene["intrinsics"]

    if NUM_STEPS <= 0:
        return {
            "depth": init_depth.detach(),
            "poses": init_poses.detach(),
            "valid_mask": torch.ones_like(init_depth, dtype=torch.bool),
        }

    t0 = time.time()
    sem = lift_semantics_3d(scene)

    rgb_wx, rgb_wy = rgb_edge_weights(rgb, sigma=RGB_EDGE_SIGMA)
    if scene["mode"] != "none" and W_SEM_BOUNDARY > 0:
        wx, wy = semantic_edge_weights(sem["instance_id"], sem["boundary_prob"], rgb_wx, rgb_wy)
    else:
        wx, wy = rgb_wx, rgb_wy

    model = RefinementModel(init_depth, init_poses, refine_poses=(LR_POSE > 0))
    param_groups = [{"params": [model.delta_log_depth], "lr": LR_DEPTH, "initial_lr": LR_DEPTH}]
    if LR_POSE > 0:
        param_groups.append({"params": [model.delta_rot, model.delta_trans], "lr": LR_POSE, "initial_lr": LR_POSE})
    optimizer = torch.optim.Adam(param_groups)

    for step in range(NUM_STEPS):
        elapsed = time.time() - t0
        if elapsed >= time_budget_sec * 0.92:
            break

        lrm = get_lr_mult(step / max(NUM_STEPS - 1, 1))
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm

        optimizer.zero_grad(set_to_none=True)
        depth, poses, _ = model()

        # 1. Confidence-weighted data fidelity to Déjà View
        loss_data = (F.huber_loss(depth, init_depth, reduction="none", delta=HUBER_DELTA) * init_conf).mean()

        # 2. Multi-view geometric reprojection consistency
        loss_mv = multiview_reprojection_loss(depth, poses, K, init_conf) if W_MULTIVIEW > 0 else depth.new_tensor(0.0)

        # 3. 2nd-order edge-aware curvature smoothness
        loss_curv = curvature_smoothness_loss(depth, wx, wy) if W_CURVATURE > 0 else depth.new_tensor(0.0)

        # 4. Edge-aware surface normal smoothness
        if W_NORMAL_SMOOTH > 0:
            nrm = compute_depth_normals(depth, K)
            dn_x = (1.0 - (nrm[:, :, 1:, :] * nrm[:, :, :-1, :]).sum(dim=-1)).clamp_min(0.0)
            dn_y = (1.0 - (nrm[:, 1:, :, :] * nrm[:, :-1, :, :]).sum(dim=-1)).clamp_min(0.0)
            loss_nrm = (wx * dn_x).mean() + (wy * dn_y).mean()
        else:
            loss_nrm = depth.new_tensor(0.0)

        # 5. Optional semantic planar prior (Track B & C)
        if scene["mode"] != "none" and W_PLANAR > 0:
            pts_w = unproject_depth(depth, K, poses)
            loss_planar = planar_prior_loss(pts_w, sem["struct_id"], sem["instance_id"], init_conf)
        else:
            loss_planar = depth.new_tensor(0.0)

        loss = (
            W_DATA * loss_data
            + W_MULTIVIEW * loss_mv
            + W_CURVATURE * loss_curv
            + W_NORMAL_SMOOTH * loss_nrm
            + W_PLANAR * loss_planar
        )

        if not torch.isfinite(loss):
            print(f"FAIL: non-finite loss at step {step} on {scene['scene_id']}")
            raise SystemExit(1)

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        depth_out, poses_out, mask_out = model()
    return {"depth": depth_out.detach(), "poses": poses_out.detach(), "valid_mask": mask_out.detach()}


# ---------------------------------------------------------------------------
# Benchmark Entry Point & Standardized Summary Output
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch 3D post-hoc refinement run")
    parser.add_argument("--mode", type=str, default=MODE, choices=("none", "oracle", "predicted"))
    parser.add_argument("--split", type=str, default=SPLIT, choices=("dev", "val", "test"))
    parser.add_argument("--k", type=int, default=DVLT_K)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visual-audit", type=str, default=None, help="Optional dir to save visual audit PNGs")
    parser.add_argument("--allow-test", action="store_true", help="Unlock blind test split (ONLY for final eval)")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(42)
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    dev_str = "cuda" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        torch.set_float32_matmul_precision("high")

    scene_ids = get_split_scenes(args.split, allow_test=args.allow_test)
    per_scene_budget = TIME_BUDGET / max(len(scene_ids), 1)
    predictions: dict[str, dict[str, Any]] = {}
    refine_times, sem_times = [], []

    for sid in scene_ids:
        scene = load_scene(sid, mode=args.mode, dvlt_k=args.k, device=dev_str)
        sem_times.append(scene["sem_extract_sec"])
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        out = refine_scene(scene, time_budget_sec=per_scene_budget)
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0
        refine_times.append(dt)
        predictions[sid] = {
            "depth": out["depth"].detach().cpu().numpy().astype(np.float32),
            "poses": out["poses"].detach().cpu().numpy().astype(np.float32),
            "valid_mask": out["valid_mask"].detach().cpu().numpy().astype(bool) if out.get("valid_mask") is not None else None,
        }
        print(f"scene {sid} | refine: {dt:.2f}s | sem_extract: {scene['sem_extract_sec']:.2f}s")

    metrics = evaluate_reconstruction(predictions, split=args.split, dvlt_k=args.k, allow_test=args.allow_test)
    if args.visual_audit:
        save_visual_audit(predictions, split=args.split, dvlt_k=args.k, out_dir=args.visual_audit, allow_test=args.allow_test)

    total_refine_sec = float(np.sum(refine_times))
    mean_refine_sec = float(np.mean(refine_times))
    mean_sem_sec = float(np.mean(sem_times))
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) if use_cuda else 0.0

    print("---")
    print(f"val_score:        {metrics['val_score']:.6f}")
    print(f"cd_cm:            {metrics['cd_cm']:.4f}")
    print(f"acc_cm:           {metrics['acc_cm']:.4f}")
    print(f"comp_cm:          {metrics['comp_cm']:.4f}")
    print(f"f1_2cm:           {metrics['f1_2cm']:.4f}")
    print(f"f1_5cm:           {metrics['f1_5cm']:.4f}")
    print(f"nae_deg:          {metrics['nae_deg']:.2f}")
    print(f"edge_err_cm:      {metrics['edge_err_cm']:.4f}")
    print(f"coverage:         {metrics['coverage']:.4f}")
    print(f"refine_seconds:   {total_refine_sec:.1f}")
    print(f"scene_seconds:    {mean_refine_sec:.2f}")
    print(f"sem_extract_sec:  {mean_sem_sec:.2f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mode:             {args.mode}")
    print(f"split:            {args.split}")
    print(f"dvlt_k:           {args.k}")


if __name__ == "__main__":
    main()
