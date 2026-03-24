# OpenVLA-OFT Reproduction Report

OpenVLA-OFT (7B, L1 regression action head with parallel decoding) evaluated on LIBERO-Spatial.

---

## Model Info

| Field | Value |
|-------|-------|
| **Model** | OpenVLA-OFT (7B) — `moojink/openvla-7b-oft-finetuned-libero-spatial` |
| **Architecture** | OpenVLA 7B + L1 regression action head, action chunking (8-step chunks), parallel decoding (26× faster than OpenVLA) |
| **Loading** | `get_vla(cfg)` from openvla-oft repo (`e4287e9`) with `use_proprio=True`, `center_crop=True` |

### Common Code Modifications

**Gripper inversion (critical)**: The OpenVLA family outputs gripper values in [0,1] RLDS convention (0=close, 1=open). The harness's LIBERO benchmark expects robosuite convention (-1=open, +1=close). Applied `-sign(2x - 1)` transformation in `oft.py` to convert correctly. Without this fix: 0% success. Simple negation (`*-1`) was also wrong because `close=0.1` → `-0.1` → LIBERO interprets as "open".

**Proprioceptive state key mismatch**: LIBERO sends state as `obs["states"]` (plural) but OFT reads `obs["state"]` (singular). Fixed OFT to check both keys: `obs.get("states", obs.get("state"))`.

---

## LIBERO-Spatial

| Field | Value |
|-------|-------|
| **Status** | `complete` |
| **Date** | 2026-03-22 |
| **Harness commit** | `reproduce-libero` branch (based on `6a48e3e`) |
| **Docker image** | `ghcr.io/allenai/vla-evaluation-harness/libero:latest` (locally built) |
| **Benchmark** | LIBERO-Spatial — 10 tasks × 50 episodes = 500 episodes |
| **Hardware** | Model server: 1 × A100-PCIE-40GB (GPU 0); Benchmark: 1 × A100-PCIE-40GB (GPU 1, Docker) |
| **Seed** | 7 |
| **Action space** | 7D (6D delta pose + 1D gripper), chunk size 8 (model) / buffer 10 (server), `unnorm_key=libero_spatial_no_noops` |

### How to Reproduce

```bash
# 1. Build Docker image locally (includes harness code with HELLO handshake)
bash docker/build.sh libero

# 2. Start model server (GPU 0, needs ~14GB VRAM for 7B model)
CUDA_VISIBLE_DEVICES=0 uv run src/vla_eval/model_servers/oft.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_images_in_input 2 \
    --chunk_size 10 \
    --port 8000

# 3. Run evaluation (GPU 1 for Docker rendering)
CUDA_VISIBLE_DEVICES=1 vla-eval run \
    --config configs/libero_all_with_state.yaml \
    --yes --gpus 1
```

### Benchmark-Specific Notes

- Config requires `send_state: true` and `send_wrist_image: true` (OFT uses both proprioceptive state and wrist camera).
- Checkpoint `openvla-7b-oft-finetuned-libero-spatial` is fine-tuned for Spatial suite ONLY. Separate checkpoints needed for Object/Goal/10 suites.
- First run downloads ~14GB of model weights from HuggingFace.

### Results

| Task | Score | Reference | Diff | Verdict |
|------|:-----:|:---------:|:----:|:-------:|
| pick up bowl between plate and ramekin | **100%** | — | — | — |
| pick up bowl next to ramekin | **100%** | — | — | — |
| pick up bowl from table center | **100%** | — | — | — |
| pick up bowl on cookie box | **100%** | — | — | — |
| pick up bowl in top drawer | **96%** | — | — | — |
| pick up bowl on ramekin | **96%** | — | — | — |
| pick up bowl next to cookie box | **100%** | — | — | — |
| pick up bowl on stove | **96%** | — | — | — |
| pick up bowl next to plate | **100%** | — | — | — |
| pick up bowl on wooden cabinet | **100%** | — | — | — |
| **Overall** | **98.8%** | **97.6%** | **+1.2 pp** | **Reproduced** |

All tasks reproduced at or above reference. Overall score 98.8% vs paper 97.6% (+1.2 pp). Wall-clock: ~1.5 hours for 500 episodes (single GPU, no sharding).

### Discussion

The +1.2 pp improvement over the reference score is within expected variance for 50-episode evaluation. Three tasks scored 96% (2 failures each out of 50), all others scored 100%. The slight overperformance may be due to seed-dependent initial state selection.

Key finding: the gripper action convention mismatch between OpenVLA's RLDS training data and robosuite's execution convention is a common pitfall. The correct transformation is `gripper_robosuite = -sign(2 * gripper_rlds - 1)`, not simple negation.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-22 | Initial LIBERO-Spatial reproduction (98.8%) |
