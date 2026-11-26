# autoresearch-3d-post-hoc

Autonomous research framework for **Semantic-Guided Refinement for Feed-Forward 3D Reconstruction** (TUM Practical Course: Generative AI · WS 2026/27, Project 06), built directly on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

- **Feed-Forward 3D Backbone:** [Déjà View (`nv-tlabs/dvlt`)](https://github.com/nv-tlabs/dvlt) — 117M-param looped transformer with inference-time compute knob $K \in \{4, 8, 12, 16\}$.
- **Benchmark Dataset:** [NVIDIA SAGE-10k (`nvidia/SAGE-10k`)](https://huggingface.co/datasets/nvidia/SAGE-10k) — selectively downloaded and rendered via headless Blender (`bpy==4.3.0`) for exact metric GT depth, GT normals, GT object boundaries, and oracle semantics.
- **Target Hardware:** Single NVIDIA RTX 3080 (10–12 GB VRAM), Python 3.11 managed with `uv`.

---

## Project Structure

```text
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

- **`prepare.py`** — Fixed constants (`TIME_BUDGET = 120`, `NUM_VIEWS = 10`, `392×518`), selective SAGE-10k download (`8 dev / 4 val / 16 locked test`), headless `bpy` PBR rendering, sequential RTX 3080-safe caching of Déjà View (`K=8` or `K ∈ {4, 8, 12, 16}`) and predicted semantics (SAM 2, Mask2Former, DINOv2), mode-isolated scene loading (`none` / `oracle` / `predicted`), fixed Open3D TSDF evaluation harness, and built-in visual audit helper.
- **`train.py`** — Single file edited by the agent. Contains `RefinementModel` (log-depth residuals + anchored SE(3) pose deltas), `lift_semantics_3d` (Milestone 2 cross-view semantic lifting), differentiable 3D geometric & semantic losses, optimizer, and time-budgeted refinement loop.
- **`program.md`** — Agent instructions implementing a **Three-Track Controlled Study**:
  1. **Track A (`MODE = "none"`):** Geometry + photometric + RGB-edge control (all semantics strictly stripped in `prepare.py`).
  2. **Track B (`MODE = "oracle"`):** Exact SAGE-10k masks & structural categories (theoretical upper bound).
  3. **Track C (`MODE = "predicted"`):** Predicted SAM 2 + Mask2Former + DINOv2 + cross-view 3D semantic association.

---

## Quick Start (RTX 3080 Workstation, Python 3.11 + `uv`)

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repo and sync self-contained Python 3.11 environment
git clone https://github.com/peter-hofinger/autoresearch-3d-post-hoc.git
cd autoresearch-3d-post-hoc
uv sync

# 3. (Optional) 30-second synthetic smoke test (zero downloads)
uv run prepare.py --synthetic --num-dev 4 --num-val 2 --num-test 4
uv run train.py --visual-audit visuals/smoke

# 4. Full SAGE-10k + Déjà View + Semantics preparation (one-time)
rm -rf ~/.cache/autoresearch-3d   # clear synthetic cache first if step 3 was run
uv run huggingface-cli login
uv run prepare.py --pareto-ks     # caches K in {4, 8, 12, 16} for Milestone 4 Pareto curve

# 5. Manually verify raw Déjà View baseline (reports val_score: 1.000000)
uv run train.py
```

---

## Running Autonomous Research

Launch Antigravity in this repo on your workstation and prompt:

```text
Have a look at program.md, prepare.py, and train.py. Let's set up Track A (no-semantics control) first, verify the baseline, and kick off the autonomous loop.
```

---

## Evaluation Metric (`val_score`)

Predictions are aligned using the **frozen raw-DVLT Sim(3) transform** (zero alignment gaming) and fused via **fixed `1.5 cm` Open3D TSDF** against exact GT geometry across four orthogonal axes (raw Déjà View K=8 = `1.000000`, lower is better):

$$\text{val\_score} = 0.30 \cdot \frac{\text{CD}}{\text{CD}_0} + 0.25 \cdot \frac{1 - \text{F1}_{2\text{cm}}}{1 - \text{F1}_{2\text{cm},0}} + 0.25 \cdot \frac{\text{NAE}}{\text{NAE}_0} + 0.20 \cdot \frac{\text{EdgeErr}}{\text{EdgeErr}_0}$$

- **`CD`**: Chamfer Distance (Accuracy + Completeness, cm)
- **`F1@2cm`**: Surface F-score @ 2 cm (prevents point deletion cheating)
- **`NAE`**: Mean Normal Angular Error vs. exact GT normals (degrees)
- **`EdgeErr`**: Depth error strictly on GT object silhouettes & depth discontinuities (cm; prevents oversmoothing)
- **Guardrails:** `coverage < 0.95` incurs a heavy penalty; total refinement time is bounded by `TIME_BUDGET = 120s`.

---

## Final Blind Test Evaluation

Once all three tracks converge, evaluate each winning branch **once** on the locked blind `test` split:

```bash
uv run train.py --mode none      --split test --allow-test --visual-audit visuals/test_none
uv run train.py --mode oracle    --split test --allow-test --visual-audit visuals/test_oracle
uv run train.py --mode predicted --split test --allow-test --visual-audit visuals/test_pred
```

## License

MIT
