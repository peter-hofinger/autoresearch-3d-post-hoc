# autoresearch-3d-post-hoc

This is an autonomous research setup for **Semantic-Guided Refinement for Feed-Forward 3D Reconstruction** (Déjà View / DVLT on NVIDIA SAGE-10k, single RTX 3080).

Core scientific question:
> *Does semantic guidance (instance masks, object categories, dense semantic features, and 3D cross-view semantic association) measurably improve surface quality, geometric accuracy, and object boundary preservation beyond an equally-optimized non-semantic refinement baseline — and is the gain worth the extra compute?*

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag & track**: propose a tag based on today's date and track:
   - **Track A (`autoresearch/<tag>-geom`)**: Non-semantic control (`MODE = "none"` in `train.py`). *Always run Track A first* to build a genuinely strong geometry/RGB-only baseline.
   - **Track B (`autoresearch/<tag>-oracle`)**: Branch off the best Track A commit, set `MODE = "oracle"` (exact SAGE-10k masks & structural categories) to establish the theoretical semantic upper bound.
   - **Track C (`autoresearch/<tag>-pred`)**: Branch off the best Track B (or Track A) commit, set `MODE = "predicted"` (SAM 2 + Mask2Former + DINOv2) to solve cross-view 3D semantic association and real-world semantic refinement.
2. **Create the branch**: `git checkout -b autoresearch/<tag>-<track>`.
3. **Read the in-scope files**: The repo is deliberately minimal. Read them in full:
   - `README.md` — repository context.
   - `prepare.py` — frozen constants, data prep, mode isolation, TSDF surface extraction, and evaluation harness. **Do not modify.**
   - `train.py` — the single file you modify. Refinement model, semantic lifting, losses, optimizer, and refinement loop.
4. **Verify cached data exists**: Check `~/.cache/autoresearch-3d/` contains `splits.json`, scene caches (`gt.npz`, `dvlt_k8.pt`, `semantics_pred.pt`), and `baseline_dev_k8.json`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize `results.tsv`**: Create `results.tsv` (untracked by git) with the header row below.
6. **Confirm and go**: Confirm setup with the user, then enter the autonomous loop.

---

## Experimentation

Each experiment runs on a single RTX 3080 GPU under a **fixed time budget of `TIME_BUDGET = 120` seconds** across the `dev` scenes (~15s/scene). Launch simply as:

```bash
uv run train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — this is the ONLY file you edit. Everything is fair game: parameterization (depth residuals, SE(3) pose refinement, implicit/surfel/bilateral/PDE solvers), cross-view 3D semantic lifting (`lift_semantics_3d`), losses (multi-view reprojection, 2nd-order curvature, anisotropic normal smoothing, flying-pixel cleanup, soft SVD/RANSAC planarity, Manhattan/gravity orthogonality, category-dependent detail protection, DINO feature consistency), optimizers (Adam, L-BFGS, IRLS, coarse-to-fine), and schedules.

**What you CANNOT do:**
- Modify `prepare.py`. It is strictly read-only.
- Read `gt.npz`, ground-truth depths/normals/poses, or the locked `test` split from `train.py`.
- Bypass mode isolation (`MODE = "none"` strips all semantic tensors to `None`).
- Install new packages or modify `pyproject.toml`.
- Cheat metrics by pruning uncertain regions (`coverage` must stay `>= 0.95` or `val_score` is heavily penalized).

**The goal: minimize `val_score` (raw Déjà View K=8 baseline = `1.000000`).**
`prepare.py` aligns outputs with the frozen raw-DVLT Sim(3) transform, fuses a fixed `1.5 cm` Open3D TSDF surface mesh, and scores 4 orthogonal axes:
- `cd_cm` (30%): Chamfer Distance (Accuracy + Completeness)
- `1 - f1_2cm` (25%): F-score error @ 2 cm (prevents point deletion)
- `nae_deg` (25%): Mean Normal Angular Error vs. exact GT normals (measures surface smoothness & noise reduction)
- `edge_err_cm` (20%): Depth error strictly on GT object silhouettes & depth discontinuities (prevents oversmoothing & flying pixels)

**VRAM & runtime**: Keep `peak_vram_mb < 9000` (RTX 3080 headroom) and `refine_seconds <= 125.0`. If a run exceeds 4 minutes total wall-clock, kill it and treat as a failure.

**Simplicity criterion**: All else being equal, simpler and faster is better. A tiny `0.001` gain that adds 50 lines of fragile code or doubles runtime is not worth keeping. Equal or better `val_score` with simpler code is always a keep.

**First run**: Always run `train.py` as-is first on each track to log the verified starting point.

---

## Output Format & Logging

Extract metrics from `run.log`:

```bash
grep -E "^(val_score|cd_cm|f1_2cm|nae_deg|edge_err_cm|coverage|scene_seconds|peak_vram_mb):" run.log
```

Log every run to `results.tsv` (tab-separated, 11 columns — leave untracked by git):

```text
commit	track	val_score	cd_cm	f1_2cm	nae_deg	edge_err_cm	scene_sec	memory_gb	status	description
```

- `commit`: 7-char short git hash
- `track`: `none`, `oracle`, or `predicted`
- `val_score`: 6 decimals (`0.000000` for crash)
- `cd_cm`, `f1_2cm`, `nae_deg`, `edge_err_cm`, `scene_sec`, `memory_gb`: numeric metrics (`0.0` for crash; `memory_gb = peak_vram_mb / 1024`)
- `status`: `keep`, `discard`, or `crash`
- `description`: concise single-line summary (no tabs)

---

## Research Directions by Track

- **Track A (`MODE = "none"` — Strong Geometry/RGB Control)**:
  Activate `NUM_STEPS`, tune LR/schedules/robust losses (Huber/Cauchy/Charbonnier), all-pairs/top-k multi-view depth & normal reprojection consistency, 2nd-order curvature + anisotropic normal smoothing, RGB-edge weights, anchored SE(3) pose refinement (`LR_POSE > 0`), photometric SSIM/L1 consistency, and flying-pixel filtering without dropping `coverage < 0.95`.
- **Track B (`MODE = "oracle"` — Semantic Upper Bound)**:
  Zero-bleed instance-boundary edge stopping, turning OFF RGB edge stopping inside `planar_stuff` (`struct_id == 1`) so textured walls/floors smooth cleanly, soft SVD/IRLS planarity + Manhattan/gravity orthogonality for `planar_stuff`, curvature/detail preservation for `rigid_thing` (`struct_id == 2`), anti-oversmoothing protection for `thin_or_nonrigid` (`struct_id == 3`), and boundary-pixel snapping toward interior instance depth to crush `edge_err_cm`.
- **Track C (`MODE = "predicted"` — Real-World Semantics & Milestone 2 Lifting)**:
  Upgrade `lift_semantics_3d()`: cross-view 3D semantic label voting (eliminating Mask2Former flicker), cross-view SAM 2 mask association via 3D reprojection overlap + DINOv2 cosine similarity, continuous DINO affinity edge weights (graceful fallback when SAM 2 errs), cross-view DINO feature reprojection loss, `class_conf`-gated fallback to Track A geometry priors, and pruning cues that do not justify their extraction compute.

---

## The Experiment Loop

LOOP FOREVER:

1. Inspect git state and `results.tsv`; pick the highest-leverage hypothesis based on which sub-metric (`cd_cm`, `f1_2cm`, `nae_deg`, `edge_err_cm`) has the largest gap.
2. Edit `train.py`.
3. `git add train.py && git commit -m "<short description>"`
4. Run: `uv run train.py > run.log 2>&1`
5. Read metrics: `grep -E "^(val_score|cd_cm|f1_2cm|nae_deg|edge_err_cm|coverage|scene_seconds|peak_vram_mb):" run.log`
6. If empty or crashed, inspect `tail -n 50 run.log`. Fix trivial bugs immediately; if fundamentally broken, log `crash` in `results.tsv` and `git reset --hard HEAD~1`.
7. Append the row to `results.tsv` (never commit `results.tsv`).
8. If `val_score` improved AND `coverage >= 0.95` AND `refine_seconds <= 125.0` AND complexity is justified → **keep** (advance branch). On breakthroughs (`>= 0.015` improvement), optionally save visual audits via `uv run train.py --visual-audit visuals/<track> > /dev/null 2>&1`.
9. Otherwise → **discard** (`git reset --hard HEAD~1`).
10. Every ~15 keeps (or before switching tracks), run a single check on `--split val` (NEVER `--split test`) to verify generalization.

**NEVER STOP**: Once the experiment loop has begun (after setup confirmation), do NOT pause to ask the human whether to continue. Keep iterating autonomously until manually stopped.
