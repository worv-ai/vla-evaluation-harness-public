# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "starvla",
#     "torch>=2.0",
#     "torchvision>=0.17",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "opencv-python>=4.0",
#     "numpy>=1.24",
#     "accelerate",
#     "kernels>=0.11.0",
#     "qwen-vl-utils",
#     "omegaconf",
#     "rich",
#     "diffusers",
#     "timm",
#     "einops",
#     "scipy",
#     "transforms3d",
#     "huggingface-hub",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# starvla = { git = "https://github.com/starVLA/starVLA.git", rev = "eaa51c4c2f4012d42f1036ee318d41942e8f97a3" }
#
# [tool.uv]
# exclude-newer = "2026-05-08T00:00:00Z"
# ///
"""RLT model server — RL post-training of a frozen QwenGR00T VLA in sim.

RLT skeleton: a frozen VLA proposes a reference action chunk; a trainable
**residual actor**, conditioned on the VLA's action-query (aq) token embedding
(the "RL-token" readout), edits the chunk in normalized space; the policy is
refined from **sparse terminal sim success** via **advantage-weighted residual
regression (AWR)** — reinforce the (explored) residuals from
better-than-average episodes, regularized toward 0 so it stays near the base.

Why AWR (not TD3): episodes are ~120 steps with only a terminal sparse reward,
so per-step TD credit assignment barely propagates. AWR uses episode-level
advantage (success − running baseline), which is robust for this regime and
matches RLT's "good base policy, refine from what worked" spirit.

Serving stays identical to ``StarVLAModelServer`` (sim runs unchanged in the
vla-eval benchmark container). This server is the actor *and* the learner, and
**checkpoints the actor** so training persists and can be greedy-evaluated:

  * ``rl_train=true``  — explore (residual + noise), collect, learn, save actor.
  * ``rl_train=false`` — deterministic learned residual (load actor, no noise,
    no learning) = greedy eval. ``residual_scale=0`` = pure base policy.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.starvla import StarVLAModelServer
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


def _mlp(sizes: list[int]):
    import torch.nn as nn

    layers: list[Any] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class RLTModelServer(StarVLAModelServer):
    """Frozen QwenGR00T + residual actor (aq-conditioned) + AWR, sparse terminal reward."""

    def __init__(
        self,
        *,
        rl_train: bool = True,
        residual_scale: float = 0.05,
        explore_std: float = 0.03,
        base_bias: float = 0.0,
        value_gate: bool = True,
        explore_episode: bool = False,
        hidden: int = 512,
        lr: float = 3e-4,
        replay_size: int = 100_000,
        batch_size: int = 256,
        updates_per_episode: int = 64,
        warmup_episodes: int = 4,
        awr_temp: float = 0.5,
        awr_wmax: float = 20.0,
        l2_coef: float = 0.05,
        succ_ema_decay: float = 0.95,
        ckpt_dir: str | None = None,
        save_every: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        import torch

        self.rl_train = rl_train
        self.residual_scale = residual_scale
        self.explore_std = explore_std
        self.base_bias = base_bias  # diagnostic handicap: known constant shift on base position dims
        self.value_gate = value_gate  # gate residual by (1-V); disable to isolate actor recovery
        self.explore_episode = explore_episode  # one constant offset per episode (vs per-step noise) -> discovers directional corrections
        self.hidden = hidden
        self.lr = lr
        self.batch_size = batch_size
        self.updates_per_episode = updates_per_episode
        self.warmup_episodes = warmup_episodes
        self.awr_temp = awr_temp
        self.awr_wmax = awr_wmax
        self.l2_coef = l2_coef
        self.succ_ema_decay = succ_ema_decay
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.save_every = save_every

        self._rl_device = next(self._model.parameters()).device
        self._replay: deque = deque(maxlen=replay_size)
        self._ep_buf: dict[str, list] = {}
        self._ep_noise: dict[str, np.ndarray] = {}
        self._rl_built = False
        self._episodes = 0
        self._succ_ema: float | None = None
        self._last_aq = None

        # Capture the action-query embedding (RL-token): the action head gets ``aq`` first.
        orig_predict = self._model.action_model.predict_action

        def _wrapped(aq, *a, **k):
            self._last_aq = aq.detach()
            return orig_predict(aq, *a, **k)

        self._model.action_model.predict_action = _wrapped
        logger.info("RLT v2(AWR): rl_train=%s residual_scale=%s ckpt_dir=%s", rl_train, residual_scale, ckpt_dir)

    # ── actor (built lazily once dims known; loads checkpoint if present) ──
    def _ensure_rl(self, embed_dim: int, act_flat: int) -> None:
        if self._rl_built:
            return
        import torch

        self._act_flat = act_flat
        self._actor = _mlp([embed_dim, self.hidden, self.hidden, act_flat]).to(self._rl_device)
        self._value = _mlp([embed_dim, self.hidden, self.hidden, 1]).to(self._rl_device)  # per-state baseline
        self._opt = torch.optim.Adam(
            list(self._actor.parameters()) + list(self._value.parameters()), lr=self.lr
        )
        if self.ckpt_dir and (self.ckpt_dir / "actor.pt").is_file():
            sd = torch.load(self.ckpt_dir / "actor.pt", map_location=self._rl_device)
            self._actor.load_state_dict(sd["actor"])
            if "value" in sd:
                self._value.load_state_dict(sd["value"])
            self._succ_ema = sd.get("succ_ema")
            self._episodes = sd.get("episodes", 0)
            logger.info("RLT loaded actor from %s (episodes=%d succ_ema=%s)", self.ckpt_dir, self._episodes, self._succ_ema)
        self._rl_built = True
        logger.info("RLT actor+value built: embed=%d act_flat=%d hidden=%d", embed_dim, act_flat, self.hidden)

    def _save(self) -> None:
        if not self.ckpt_dir:
            return
        import torch

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self._actor.state_dict(),
                "value": self._value.state_dict(),
                "succ_ema": self._succ_ema,
                "episodes": self._episodes,
            },
            self.ckpt_dir / "actor.pt",
        )

    # ── acting: reference chunk + (learned) residual edit; record residual taken ──
    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        import cv2
        import torch
        from PIL import Image as PILImage

        def _prep(img: Any) -> PILImage.Image:
            if isinstance(img, np.ndarray):
                if self._image_size and img.shape[:2] != self._image_size:
                    img = cv2.resize(img, (self._image_size[1], self._image_size[0]), interpolation=cv2.INTER_AREA)
                return PILImage.fromarray(img).convert("RGB")
            return img

        examples = []
        for obs in obs_batch:
            src = obs.get("images", {})
            imgs = [_prep(v) for v in src.values()] if isinstance(src, dict) else [_prep(src)]
            ex: dict[str, Any] = {"image": imgs, "lang": obs.get("task_description", "")}
            st = obs.get("states", obs.get("state"))
            if st is not None:
                st = np.asarray(st, dtype=np.float32).flatten()
                if len(st) == 8:
                    st = np.concatenate([st[:6], [st[6:8].mean()]])
                ex["state"] = st.reshape(1, -1)
            examples.append(ex)

        result = self._model.predict_action(examples)  # sets self._last_aq
        norm_batch = np.asarray(result["normalized_actions"], dtype=np.float32)  # [B,T,D]
        B, T, D = norm_batch.shape
        if self.base_bias:  # diagnostic handicap on x-position (dim 0); optimal residual = -base_bias on dim 0
            norm_batch = norm_batch.copy()
            norm_batch[:, :, 0] += self.base_bias
        aq = self._last_aq.float().mean(dim=1)  # [B,H]

        residual = np.zeros((B, T * D), dtype=np.float32)
        gate = np.ones((B, 1), dtype=np.float32)
        if self.residual_scale > 0:
            self._ensure_rl(aq.shape[-1], T * D)
            with torch.no_grad():
                residual = (self._actor(aq).tanh() * self.residual_scale).cpu().numpy()  # [B,T*D]
                if self.value_gate:
                    gate = (1.0 - self._value(aq)).clamp(0.0, 1.0).cpu().numpy()  # [B,1]; ~0 where base already wins

        outputs = []
        for i in range(B):
            r = residual[i].copy()
            if self.rl_train and self.residual_scale > 0:
                sid = ctx_batch[i].session_id
                if self.explore_episode:  # one constant offset per episode -> shifts the whole trajectory
                    n = self._ep_noise.get(sid)
                    if n is None:
                        n = np.random.normal(0, self.explore_std, r.shape).astype(np.float32)
                        self._ep_noise[sid] = n
                else:
                    n = np.random.normal(0, self.explore_std, r.shape).astype(np.float32)
                r = r + n
                self._ep_buf.setdefault(sid, []).append(
                    (aq[i].detach().cpu().numpy().astype(np.float32), r.astype(np.float32))
                )
            r_exec = r * float(gate[i, 0])  # value-gate: suppress residual on states the base already solves (do-no-harm)
            a_norm = np.clip(norm_batch[i] + r_exec.reshape(T, D), -1.0, 1.0)
            outputs.append({"actions": self._to_env_action(a_norm, ctx_batch[i])})
        return outputs

    def _to_env_action(self, a_norm: np.ndarray, ctx: SessionContext) -> np.ndarray:
        from transforms3d.euler import euler2axangle

        actions = self._unnormalize(a_norm.copy())
        if self.gripper_invert:
            actions[:, 6] = 1.0 - 2.0 * actions[:, 6]
        ens = self._ensemblers.get(ctx.session_id)
        if ens is not None:
            actions = ens(actions)[np.newaxis]
        if self.euler_to_axisangle and actions.shape[-1] >= 6:
            for t in range(actions.shape[0]):
                axis, angle = euler2axangle(actions[t, 3], actions[t, 4], actions[t, 5])
                actions[t, 3:6] = axis * angle
        return actions

    # ── learning: episode-level advantage → AWR on the residuals taken ──
    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        buf = self._ep_buf.pop(ctx.session_id, [])
        self._ep_noise.pop(ctx.session_id, None)
        if self.rl_train and self.residual_scale > 0 and buf:
            metrics = result.get("metrics", result) if isinstance(result, dict) else {}
            success = float(bool(metrics.get("success", result.get("success", False))))
            adv_log = success - (self._succ_ema if self._succ_ema is not None else success)
            self._succ_ema = success if self._succ_ema is None else (
                self.succ_ema_decay * self._succ_ema + (1 - self.succ_ema_decay) * success
            )
            for aq, r in buf:
                self._replay.append((aq, r, np.float32(success)))  # store return; baseline learned by V(aq)
            self._episodes += 1
            loss = None
            if self._episodes >= self.warmup_episodes and len(self._replay) >= self.batch_size:
                loss = float(np.mean([self._awr_update() for _ in range(self.updates_per_episode)]))
            if self.ckpt_dir and self._episodes % self.save_every == 0:
                self._save()
            logger.info(
                "RLT ep=%d success=%.0f adv=%+.3f succ_ema=%.3f |replay|=%d awr_loss=%s",
                self._episodes, success, adv_log, self._succ_ema or 0.0, len(self._replay),
                f"{loss:.4f}" if loss is not None else "warmup",
            )
        await super().on_episode_end(result, ctx)

    def _awr_update(self):
        import torch
        import torch.nn.functional as F

        batch = random.sample(self._replay, self.batch_size)
        dev = self._rl_device
        aq = torch.as_tensor(np.stack([b[0] for b in batch]), device=dev)
        rtaken = torch.as_tensor(np.stack([b[1] for b in batch]), device=dev)
        ret = torch.as_tensor(np.stack([b[2] for b in batch]), device=dev).unsqueeze(1)

        v = self._value(aq)
        value_loss = F.mse_loss(v, ret)
        adv = ret - v.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)  # per-state advantage, normalized
        w = torch.clamp(torch.exp(adv / self.awr_temp), max=self.awr_wmax)
        pred = self._actor(aq).tanh() * self.residual_scale
        # AWR: reinforce residuals beating the state's value baseline; L2 keeps near base (do-no-harm)
        actor_loss = (w * (pred - rtaken).pow(2).mean(dim=1, keepdim=True)).mean() + self.l2_coef * pred.pow(2).mean()
        loss = actor_loss + value_loss
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        return float(actor_loss.detach())


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(RLTModelServer)
