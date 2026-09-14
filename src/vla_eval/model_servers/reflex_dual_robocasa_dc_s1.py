"""S1-only base-policy model server for the robocasa_dc benchmark (decision-gate eval).

Sibling of ``reflex_dual_robocasa_dc.py`` that loads ``ReFlExS1VLA`` (GR00T-N1.5 flow head, NO
System2) and predicts via ``ReFlExS1GR00T.predict_action`` with an empty K=0 cognition — the base
policy alone, no GR-1 demo, no cognition cache. Obs/proprio/action handling and the
``PredictModelServer`` ws protocol are reused verbatim from the Dual server, so the Docker shards
connect unchanged. ``--cognition-off``/``--cognition-on`` are accepted as no-ops (always base).

Launch from reflex-train's ``.venv``::

    CUDA_VISIBLE_DEVICES=0 reflex-train/.venv/bin/python \\
        -m vla_eval.model_servers.reflex_dual_robocasa_dc_s1 \\
        --args.ckpt_dir <s1-only ckpt> --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import DimSpec
from vla_eval.types import Action, Observation

# Reuse the Dual server's protocol constants + PIL helper (single source of truth).
from vla_eval.model_servers.reflex_dual_robocasa_dc import (
    ACTION_COMPONENT_DIMS,
    ACTION_DIM,
    VIDEO_KEYS,
    _CHUNK_SIZE,
    _to_pil,
)

logger = logging.getLogger(__name__)


class ReFlExS1RoboCasaDCServer(PredictModelServer):
    """Serve a trained ``ReFlExS1VLA`` (base policy, no System2) for the robocasa_dc eval."""

    def __init__(
        self,
        ckpt_dir: str,
        data_dir: str = "data/robocasa_dc",  # ignored (no demo needed); kept for CLI/yaml parity
        cognition_off: bool = False,  # no-op: base policy is always cognition-free
        device: str = "cuda",
        max_batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=_CHUNK_SIZE, max_batch_size=max_batch_size)
        self.device = device

        from reflex.data.normalize import NormStats
        from reflex.models.gr00t_n15.collator import _TOKENS_PER_TILE
        from reflex.models.gr00t_n15.loading import load_eagle_tokenizer
        from reflex.models.s1.configuration_s1 import ReFlExS1Config
        from reflex.models.s1.modeling_s1 import ReFlExS1VLA
        from safetensors.torch import load_file

        cfg = ReFlExS1Config.from_pretrained(ckpt_dir)
        self.cfg = cfg
        model = ReFlExS1VLA(cfg)
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
        self.cognition_dim = cfg.cognition_dim
        self._fixed_noise = bool(os.environ.get("REFLEX_FIXED_NOISE"))
        self._noise_seeded: set[str] = set()
        self._batch_calls = 0
        self._batch_obs = 0
        # Match the Dual server's ready substring so the launcher's `ready (cognition_off=` grep works.
        logger.info(
            "ReFlExS1RoboCasaDCServer ready (cognition_off=N/A base-policy, max_batch_size=%d)", max_batch_size
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

        # Empty (K=0) cognition — ReFlExS1GR00T._backbone_features skips injection entirely.
        cog = torch.zeros(n, 0, self.cognition_dim, device=self.device, dtype=torch.bfloat16)

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

    # --- spec (identical to the Dual server) ---

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

    run_server(ReFlExS1RoboCasaDCServer)
