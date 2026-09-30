"""Reflex dual-VLA (Qwen3.5 S2 + ReFlExS1GR00T) model server for the robocasa_dc benchmark.

Ports the validated ``scripts/serve_robocasa_dc.py`` inference into a ``PredictModelServer``
so vla-eval owns the parallelism (GPU batching across sessions + episode sharding). The
GR-1 demo conditioning lives here: each session's first inference loads that episode's demo
(``obs["task_name"]`` + ``obs["episode_idx"]`` = the seed = the demo index), runs the S2 to a
cognition tensor, and caches it for the rest of the episode. ``--cognition-off`` swaps in
``null_cognition`` (the S1-alone control for the gating ablation).

Launch from reflex-train's ``.venv`` (it has both ``reflex`` and ``vla_eval``)::

    CUDA_VISIBLE_DEVICES=0 reflex-train/.venv/bin/python \\
        -m vla_eval.model_servers.reflex_dual_robocasa_dc \\
        --ckpt-dir <ckpt> --data-dir <robocasa_dc data root> --port 8000 [--cognition-off]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

# Server reads these obs.images names (POLICY_CAMERAS order: agentview_left/right + eye_in_hand).
VIDEO_KEYS = ["video.ego_view", "video.ego_view_right", "video.hand_view"]
# 6 action components, dims summing to 12 (checked against norm_stats.q01 at load).
ACTION_COMPONENT_DIMS = [3, 3, 1, 3, 1, 1]
ACTION_DIM = 12
_CHUNK_SIZE = 16  # action_horizon the policy emits; framework serves one action per env step.
_DEMO_CHUNK = 100  # LeRobot chunks_size: episodes 0..99 -> chunk-000.


def _to_pil(arr: np.ndarray) -> Image.Image:
    """HWC uint8 (the benchmark already flipped it) -> RGB PIL; squeeze stray leading dims."""
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[0]
    return Image.fromarray(np.ascontiguousarray(arr).astype(np.uint8), mode="RGB")


class ReFlExDualRoboCasaDCServer(PredictModelServer):
    """Serve a trained ``ReFlExDualVLA`` for the robocasa_dc cross-embodiment eval."""

    def __init__(
        self,
        ckpt_dir: str,
        data_dir: str = "data/robocasa_dc",
        cognition_off: bool = False,
        device: str = "cuda",
        max_batch_size: int = 8,
        demo_subdir: str = "eval_set_gr1",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=_CHUNK_SIZE, max_batch_size=max_batch_size)
        self.cognition_off = cognition_off
        self.device = device

        from reflex.data.normalize import NormStats
        from reflex.models.reflex_dual.configuration_reflex_dual import ReFlExDualConfig
        from reflex.models.gr00t_n15.collator import _TOKENS_PER_TILE
        from reflex.models.gr00t_n15.loading import load_eagle_tokenizer
        from reflex.models.reflex_dual.modeling_reflex_dual import ReFlExDualVLA
        from safetensors.torch import load_file
        from transformers import AutoProcessor

        data_root = Path(data_dir)
        # Faithful eval conditions on the HELD-OUT eval demos (one GR-1 demo per eval seed),
        # not the training demos. demo_subdir is configurable; eval_set_gr1 is the eval default.
        self.gr1_root = data_root / demo_subdir
        with open(data_root / "video_key_map_gr1.json") as f:
            self.demo_cam: dict[str, str] = json.load(f)

        cfg = ReFlExDualConfig.from_pretrained(ckpt_dir)
        self.cfg = cfg
        model = ReFlExDualVLA(cfg)
        missing, unexpected = model.load_state_dict(
            load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=False
        )
        real_missing = [k for k in missing if "eagle_linear" not in k]
        if real_missing:
            logger.warning("[load] %d missing keys (showing 8): %s", len(real_missing), real_missing[:8])
        if unexpected:
            logger.warning("[load] %d unexpected keys (showing 8): %s", len(unexpected), list(unexpected)[:8])
        model.to(device, torch.bfloat16).eval()
        self.model = model

        self.norm_stats = NormStats.from_dict(json.load(open(os.path.join(ckpt_dir, "norm_stats.json"))))
        assert sum(ACTION_COMPONENT_DIMS) == len(self.norm_stats.q01) == ACTION_DIM, (
            f"action dim {sum(ACTION_COMPONENT_DIMS)} != norm q01 len {len(self.norm_stats.q01)}"
        )

        self.eagle_tokenizer = load_eagle_tokenizer()
        self.pad_to = len(VIDEO_KEYS) * _TOKENS_PER_TILE + 128
        self.s2_processor = AutoProcessor.from_pretrained(cfg.s2.vlm_model_id)

        # Per-episode cognition cache: ctx.episode_id -> (K, H) on device. Computed once per
        # episode (demo conditioning is fixed); reused across the ~max_steps/16 predict calls.
        self._cog_cache: dict[str, torch.Tensor] = {}
        self._demo_idx_cache: dict[str, list[int]] = {}
        self._demo_idx_set_cache: dict[str, set[int]] = {}
        # DEBUG: REFLEX_FIXED_NOISE seeds the flow head per episode (by episode_idx) so rollouts
        # are deterministic and byte-comparable against the serial serve_robocasa_dc path.
        self._fixed_noise = bool(os.environ.get("REFLEX_FIXED_NOISE"))
        self._noise_seeded: set[str] = set()
        self._batch_calls = 0  # server-side batching telemetry (avg batch size = GPU-batch utilization)
        self._batch_obs = 0
        logger.info(
            "ReFlExDualRoboCasaDCServer ready (cognition_off=%s, max_batch_size=%d)", cognition_off, max_batch_size
        )

    # --- inference ---

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return self._infer([obs], [ctx])[0]

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        return self._infer(obs_batch, ctx_batch)

    @torch.no_grad()
    def _infer(self, obs_list: list[Observation], ctx_list: list[SessionContext]) -> list[Action]:
        from reflex.data.normalize import normalize_state, unnormalize
        from reflex.models.gr00t_n15.collator import build_eagle_inputs

        n = len(obs_list)
        self._batch_calls += 1
        self._batch_obs += n
        if self._batch_calls % 200 == 0:
            logger.info(
                "[batchstat] calls=%d cumulative_avg_batch=%.2f last=%d",
                self._batch_calls,
                self._batch_obs / self._batch_calls,
                n,
            )
        tasks = [str(o.get("task_description", "") or "") for o in obs_list]

        states = np.stack([np.asarray(o["state"], dtype=np.float32).reshape(-1) for o in obs_list])  # (N,53)
        proprio = torch.as_tensor(normalize_state(states, self.norm_stats), dtype=torch.bfloat16, device=self.device)

        cams_per = [[_to_pil(o["images"][k]) for k in VIDEO_KEYS] for o in obs_list]
        eis = [build_eagle_inputs(cams_per[r], tasks[r], self.eagle_tokenizer, pad_to=self.pad_to) for r in range(n)]
        eagle_input_ids = torch.cat([e["input_ids"] for e in eis], dim=0).to(self.device)
        eagle_attention_mask = torch.cat([e["attention_mask"] for e in eis], dim=0).to(self.device)
        eagle_pixel_values = torch.stack([e["pixel_values"] for e in eis], dim=0).to(self.device, torch.bfloat16)

        K = self.cfg.s2.num_cognition_tokens
        H = self.model.system2.hidden_size
        if self.cognition_off:
            cog = self.model.null_cognition.to(self.device, torch.bfloat16).view(1, K, H).expand(n, K, H)
        else:
            rows = []
            for r in range(n):
                eid = ctx_list[r].episode_id
                if eid not in self._cog_cache:
                    tn, ei = str(obs_list[r]["task_name"]), int(obs_list[r]["episode_idx"])
                    demo_pils = self._load_demo_pils(tn, ei)
                    cog_r = self._compute_cognition(tasks[r], cams_per[r], demo_pils)[0]  # (K,H)
                    self._cog_cache[eid] = cog_r
                    logger.info(
                        "[dbg] ep=%s task=%s demo_idx=%d demo=%s n_demo=%d cog.norm=%.4f text=%r proprio[:4]=%s",
                        eid[:8],
                        tn,
                        ei,
                        self._gr1_video(tn, ei).name,
                        len(demo_pils),
                        cog_r.float().norm().item(),
                        tasks[r],
                        np.round(states[r][:4], 4).tolist(),
                    )
                rows.append(self._cog_cache[eid])
            cog = torch.stack(rows, dim=0)  # (N,K,H)

        if self._fixed_noise:
            eid0 = ctx_list[0].episode_id
            if eid0 not in self._noise_seeded:
                torch.manual_seed(int(obs_list[0]["episode_idx"]))
                self._noise_seeded.add(eid0)

        action_pred = self.model.system1.predict_action(
            images=None,
            cognition=cog,
            age_ms=torch.zeros(n, device=self.device, dtype=torch.bfloat16),
            proprio=proprio,
            vlm_input_ids=eagle_input_ids,
            vlm_attention_mask=eagle_attention_mask,
            vlm_pixel_values=eagle_pixel_values,
        )  # (N, 16, 32)
        a12 = unnormalize(action_pred[:, :, :ACTION_DIM].float().cpu().numpy(), self.norm_stats)  # (N,16,12)
        return [{"actions": a12[r].astype(np.float32)} for r in range(n)]

    # --- demo + cognition (S2) ---

    def _demo_indices(self, task_name: str) -> list[int]:
        """Sorted available demo episode indices for a task (cached). Some eval tasks have
        fewer demos than eval seeds and the numbering may have gaps, so resolve against the
        actual files rather than assuming a contiguous 0..N range."""
        if task_name not in self._demo_idx_cache:
            cam = self.demo_cam[task_name]
            cam_dir = self.gr1_root / task_name / "videos"
            idxs = sorted(
                int(p.stem.split("_")[-1]) for p in cam_dir.glob(f"chunk-*/observation.images.{cam}/episode_*.mp4")
            )
            if not idxs:
                raise FileNotFoundError(f"no demo videos under {cam_dir} for cam {cam}")
            self._demo_idx_cache[task_name] = idxs
        return self._demo_idx_cache[task_name]

    def _gr1_video(self, task_name: str, ep: int) -> Path:
        cam = self.demo_cam[task_name]
        avail = self._demo_indices(task_name)
        # Faithful: eval seed == demo index. For the few eval seeds beyond a task's demo count
        # (e.g. PnPCounterToMicrowave has 40 demos for 50 seeds), wrap to a valid demo.
        demo = ep if ep in self._demo_idx_set(task_name) else avail[ep % len(avail)]
        chunk = demo // _DEMO_CHUNK
        return (
            self.gr1_root
            / task_name
            / "videos"
            / f"chunk-{chunk:03d}"
            / f"observation.images.{cam}"
            / f"episode_{demo:06d}.mp4"
        )

    def _demo_idx_set(self, task_name: str) -> set[int]:
        if task_name not in self._demo_idx_set_cache:
            self._demo_idx_set_cache[task_name] = set(self._demo_indices(task_name))
        return self._demo_idx_set_cache[task_name]

    def _load_demo_pils(self, task_name: str, ep: int) -> list[Image.Image]:
        import av

        path = str(self._gr1_video(task_name, ep))
        frames: list[np.ndarray] = []
        with av.open(path) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise RuntimeError(f"no frames decoded from demo video {path}")
        demo_fps = float(self.cfg.s2.demo_fps)
        stride = max(1, int(round(20.0 / demo_fps))) if demo_fps > 0 else 1
        idxs = list(range(0, len(frames), stride))
        max_frames = self.cfg.s2.max_demo_frames
        if max_frames > 0 and len(idxs) > max_frames:
            keep = np.linspace(0, len(idxs) - 1, max_frames, dtype=int)
            idxs = [idxs[i] for i in keep]
        return [Image.fromarray(frames[i], mode="RGB") for i in idxs]

    @torch.no_grad()
    def _compute_cognition(self, task: str, cams: list[Image.Image], demo_pils: list[Image.Image]) -> torch.Tensor:
        from reflex.models.reflex_dual.build_s2_input import build_s2_sequence

        s2_seq = build_s2_sequence(
            self.s2_processor,
            task,
            s2_images=[cams],
            demo_images=demo_pils,
            subgoals=None,
            num_cognition_tokens=self.cfg.s2.num_cognition_tokens,
        )
        dev = self.device
        cognition, _ = self.model.system2(
            input_ids=s2_seq["input_ids"].unsqueeze(0).to(dev),
            attention_mask=s2_seq["attention_mask"].unsqueeze(0).to(dev),
            pixel_values=s2_seq["pixel_values"].to(dev, torch.bfloat16),
            image_grid_thw=s2_seq["image_grid_thw"].to(dev),
            cognition_mask=s2_seq["cognition_mask"].unsqueeze(0).to(dev),
            s2_query_ids=s2_seq["s2_query_ids"].unsqueeze(0).to(dev),
            mm_token_type_ids=s2_seq["mm_token_type_ids"].unsqueeze(0).to(dev),
            s2_num_queries=1,
            s2_total_queries=1,
        )  # (B, T=1, K, H)
        return cognition[:, 0]  # (B, K, H)

    # --- lifecycle ---

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._cog_cache.pop(ctx.episode_id, None)
        await super().on_episode_end(result, ctx)

    # --- spec ---

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "actions": DimSpec(
                name="panda_omron_action",
                dims=ACTION_DIM,
                format="arm_pos3_rot3_grip1_base3_torso1_mode1",
                description="16-step chunk of the PandaOmron 12-D action.",
            )
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        from vla_eval.specs import IMAGE_RGB, LANGUAGE

        spec: dict[str, DimSpec] = {f"images.{v}": IMAGE_RGB for v in VIDEO_KEYS}
        spec["state"] = DimSpec(
            name="robot_obs", dims=53, format="robocasa_dc_53d", description="12 components concat"
        )
        spec["task_description"] = LANGUAGE
        return spec


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(ReFlExDualRoboCasaDCServer)
