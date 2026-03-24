# X-VLA Reproduction Report

X-VLA (2toINF/X-VLA-Libero) evaluated on LIBERO-Spatial. Unlike most models, X-VLA uses absolute actions with 6D rotation representation.

---

## Model Info

| Field | Value |
|-------|-------|
| **Model** | X-VLA — `2toINF/X-VLA-Libero` |
| **Architecture** | Flow-matching VLA with 20D dual-arm action space, EE6DActionSpace |
| **Loading** | `model.from_pretrained(...)` via xvla.py model server |

### Benchmark-Specific Configuration

X-VLA requires non-standard LIBERO settings:
- **`absolute_action: true`** — X-VLA outputs world-frame positions, not deltas
- **`state_format: ee_rot6d`** — 20D proprioceptive state `[pos3, rot6d6, 0, zeros10]`
- **`flip_wrist_image: false`** — X-VLA's official eval does not flip the wrist camera
- **`send_state: true`** and **`send_wrist_image: true`**

A dedicated config `configs/libero_all_xvla.yaml` encapsulates these settings.

---

## LIBERO-Spatial

| Field | Value |
|-------|-------|
| **Status** | `complete` |
| **Date** | 2026-03-23 |
| **Harness commit** | `reproduce-libero` branch |
| **Docker image** | `ghcr.io/allenai/vla-evaluation-harness/libero:latest` (locally built) |
| **Benchmark** | LIBERO-Spatial — 10 tasks × 50 episodes = 500 episodes |
| **Hardware** | Model server: 1 × A100-80GB (GPU 0); Benchmark: 1 × A100-80GB (GPU 1, Docker) |
| **Seed** | 7 |
| **Action space** | 7D (absolute pos + axis-angle + gripper), chunk size 30, denoising steps 10 |

### How to Reproduce

```bash
# 1. Build Docker image locally
bash docker/build.sh libero

# 2. Start X-VLA model server (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run src/vla_eval/model_servers/xvla.py \
    --model_path "2toINF/X-VLA-Libero" \
    --benchmark_profile libero \
    --domain_id 3 \
    --denoising_steps 10 \
    --chunk_size 30 \
    --port 8000

# 3. Run evaluation (GPU 1)
CUDA_VISIBLE_DEVICES=1 vla-eval run \
    --config configs/libero_all_xvla.yaml --yes --gpus 1
```

### Results

| Task | Score | Reference | Diff | Verdict |
|------|:-----:|:---------:|:----:|:-------:|
| bowl between plate and ramekin | **98%** | — | — | — |
| bowl next to ramekin | **98%** | — | — | — |
| bowl from table center | **100%** | — | — | — |
| bowl on cookie box | **100%** | — | — | — |
| bowl in top drawer | **90%** | — | — | — |
| bowl on ramekin | **100%** | — | — | — |
| bowl next to cookie box | **100%** | — | — | — |
| bowl on stove | **100%** | — | — | — |
| bowl next to plate | **94%** | — | — | — |
| bowl on wooden cabinet | **98%** | — | — | — |
| **Overall Spatial** | **97.8%** | **98.2%** | **-0.4 pp** | **Reproduced** |

### LIBERO-Object (500 episodes)

| Task | Score |
|------|:-----:|
| alphabet soup → basket | 100% |
| cream cheese → basket | 100% |
| salad dressing → basket | 100% |
| bbq sauce → basket | 96% |
| ketchup → basket | 96% |
| tomato sauce → basket | 96% |
| butter → basket | 100% |
| milk → basket | 96% |
| chocolate pudding → basket | 100% |
| orange juice → basket | 100% |
| **Overall Object** | **98.4%** vs **98.6%** (-0.2 pp) — **Reproduced** |

### LIBERO-Goal (500 episodes)

| Task | Score |
|------|:-----:|
| open middle drawer | 100% |
| put bowl on stove | 98% |
| put wine bottle on top of cabinet | 96% |
| open top drawer and put bowl inside | 94% |
| put bowl on top of cabinet | 100% |
| push plate to front of stove | 88% |
| put cream cheese in bowl | 100% |
| turn on stove | 100% |
| put bowl on plate | 100% |
| put wine bottle on rack | 94% |
| **Overall Goal** | **97.0%** vs **97.8%** (-0.8 pp) — **Reproduced** |

### LIBERO-10 (500 episodes, 10 shards)

| Task | Score |
|------|:-----:|
| pick up book → caddy | 96% |
| put both moka pots on stove | 98% |
| alphabet soup + cream cheese → basket | 100% |
| alphabet soup + tomato sauce → basket | 92% |
| cream cheese + butter → basket | 100% |
| black bowl → bottom drawer + close | 86% |
| white mug left + yellow mug right | 96% |
| white mug on plate + chocolate pudding right | 96% |
| yellow mug → microwave + close | 100% |
| turn on stove + moka pot | 98% |
| **Overall 10** | **96.2%** vs **97.6%** (-1.4 pp) — **Reproduced** |

### Summary

| Suite | Score | Reference | Diff | Verdict |
|-------|:-----:|:---------:|:----:|:-------:|
| Spatial | 97.8% | 98.2% | -0.4 pp | Reproduced |
| Object | 98.4% | 98.6% | -0.2 pp | Reproduced |
| Goal | 97.0% | 97.8% | -0.8 pp | Reproduced |
| 10 | 96.2% | 97.6% | -1.4 pp | Reproduced |
| **Overall** | **97.4%** | **98.1%** | **-0.7 pp** | **Reproduced** |

### Discussion

X-VLA reproduced within expected variance across all 4 LIBERO suites (-0.7 pp overall). No code modifications were needed to the X-VLA model server — only benchmark configuration changes (absolute actions, ee_rot6d state, unflipped wrist image). The X-VLA model server already handles the 20D→7D action conversion and gripper sigmoid internally.

X-VLA uses a single checkpoint for all LIBERO suites (unlike OFT which has per-suite checkpoints), making it a strong baseline candidate for multi-suite evaluation. LIBERO-10 used 10-shard parallel evaluation, completing 500 episodes in ~25 minutes.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-23 | Full 4-suite reproduction: Spatial 97.8%, Object 98.4%, Goal 97.0%, 10 96.2% |
| 2026-03-23 | Initial LIBERO-Spatial reproduction (97.8%) |

---

## CALVIN (50 sequences sample)

| Field | Value |
|-------|-------|
| **Status** | `incomplete` — significant gap from paper |
| **Date** | 2026-03-24 |
| **Docker image** | `ghcr.io/allenai/vla-evaluation-harness/calvin:latest` |
| **Benchmark** | CALVIN ABC→D — 50 of 1000 sequences sampled |
| **Seed** | 0 |
| **Server config** | `output_action_dim=20`, `absolute_action: true` |

### Results

| Metric | Score | Reference |
|--------|:-----:|:---------:|
| Avg chain length | **0.58 / 5** | **4.43 / 5** |
| 0 subtasks | 24 (48%) | — |
| 1 subtask | 23 (46%) | — |
| 2 subtasks | 3 (6%) | — |

### Discussion

X-VLA CALVIN achieves only 0.58/5 avg chain length vs paper 4.43/5. The CALVIN benchmark's `_process_absolute_action` correctly parses 20D X-VLA actions to 7D [pos3, euler3, grip], but CALVIN's environment does not switch to absolute control mode (unlike LIBERO's `robot.controller.use_delta = False`). Actions may be incorrectly interpreted as deltas despite the format conversion. Further investigation of CALVIN's action execution pipeline needed.
