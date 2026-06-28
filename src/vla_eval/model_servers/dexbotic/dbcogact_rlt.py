# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "dexbotic",
#     "torch>=2.0",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "einops",
#     "timm",
#     "sentencepiece",
#     "diffusers",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../../..", editable = true }
# dexbotic = { git = "https://github.com/MilkClouds/dexbotic.git", rev = "42f72859dfe48bb4c30a09ab151a018c2ca0700a" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""DB-CogACT RLT server — residual RL post-training on a frozen DB-CogACT (dexbotic) VLA.

Same RLT mechanism as ``starvla_rlt`` (frozen base + learned residual edit on the action chunk,
AWR with a per-state value baseline, optional best-of-N candidate selection), ported to DB-CogACT
on CALVIN, where the episode metric ``completed_subtasks`` (0..5) gives a **graded progress
return** — the dense signal a strong frozen VLA needs to actually improve.

CALVIN drives the server through the single ``predict`` path (→ ``inference_action``), so the
residual is applied there. The conditioning token ``z`` (cognition features) is captured by
wrapping the diffusion action-head's ``forward`` / ``forward_with_cfg`` (cfg_scale>1 uses the
latter, and the conditioned half is the first row of the doubled ``z``).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.types import Action, Observation
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.dexbotic.cogact import CogACTModelServer

logger = logging.getLogger(__name__)


def _mlp(sizes: list[int]):
    import torch.nn as nn

    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DBCogACTRLTServer(CogACTModelServer):
    def __init__(
        self,
        model_path: str,
        *,
        rl_train: bool = False,
        residual_scale: float = 0.05,
        explore_std: float = 0.05,
        lr: float = 1e-3,
        hidden: int = 512,
        awr_temp: float = 1.0,
        awr_wmax: float = 20.0,
        l2_coef: float = 0.01,
        succ_ema_decay: float = 0.95,
        warmup_episodes: int = 8,
        batch_size: int = 256,
        updates_per_episode: int = 32,
        replay_cap: int = 200_000,
        value_gate: bool = True,
        candidate_n: int = 0,
        candidate_std: float = 0.05,
        candidate_margin: float = 0.0,
        ckpt_dir: str | None = None,
        save_every: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path, **kwargs)  # keep cogact's cfg_scale default; forward_with_cfg handles dtype

        self.rl_train = rl_train
        self.residual_scale = residual_scale
        self.explore_std = explore_std
        self.lr = lr
        self.hidden = hidden
        self.awr_temp = awr_temp
        self.awr_wmax = awr_wmax
        self.l2_coef = l2_coef
        self.succ_ema_decay = succ_ema_decay
        self.warmup_episodes = warmup_episodes
        self.batch_size = batch_size
        self.updates_per_episode = updates_per_episode
        self.value_gate = value_gate
        self.candidate_n = candidate_n
        self.candidate_std = candidate_std
        self.candidate_margin = candidate_margin
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.save_every = save_every

        self._rl_device = self._device
        self._rl_built = False
        self._actor = self._value = self._q = self._opt = None
        self._replay: list = []
        self._replay_cap = replay_cap
        self._ep_buf: dict[str, list] = {}
        self._succ_ema: float | None = None
        self._episodes = 0
        self._last_z = None
        self._logged_scale = False

        # Capture cognition z from the diffusion action head (both cfg + non-cfg entry points).
        net = self._model.model.action_head.net

        def _make_wrap(orig):
            def _wrapped(x, t, *a, **k):
                z = k.get("z")
                if z is None and a:
                    z = a[0]
                if z is not None:
                    self._last_z = z.detach()
                return orig(x, t, *a, **k)

            return _wrapped

        net.forward = _make_wrap(net.forward)
        net.forward_with_cfg = _make_wrap(net.forward_with_cfg)
        logger.info("DB-CogACT RLT: rl_train=%s residual_scale=%s ckpt_dir=%s", rl_train, residual_scale, ckpt_dir)

    def _ensure_rl(self, embed_dim: int, act_flat: int) -> None:
        if self._rl_built:
            return
        import torch

        self._actor = _mlp([embed_dim, self.hidden, self.hidden, act_flat]).to(self._rl_device)
        self._value = _mlp([embed_dim, self.hidden, self.hidden, 1]).to(self._rl_device)
        self._q = _mlp([embed_dim + act_flat, self.hidden, self.hidden, 1]).to(self._rl_device)
        self._opt = torch.optim.Adam(
            list(self._actor.parameters()) + list(self._value.parameters()) + list(self._q.parameters()), lr=self.lr
        )
        if self.ckpt_dir and (self.ckpt_dir / "actor.pt").is_file():
            sd = torch.load(self.ckpt_dir / "actor.pt", map_location=self._rl_device)
            self._actor.load_state_dict(sd["actor"])
            if "value" in sd:
                self._value.load_state_dict(sd["value"])
            if "q" in sd:
                self._q.load_state_dict(sd["q"])
            self._succ_ema = sd.get("succ_ema")
            self._episodes = sd.get("episodes", 0)
            logger.info("DB-CogACT RLT loaded from %s (episodes=%d succ_ema=%s)", self.ckpt_dir, self._episodes, self._succ_ema)
        self._rl_built = True
        logger.info("DB-CogACT RLT built: embed=%d act_flat=%d hidden=%d", embed_dim, act_flat, self.hidden)

    def _save(self) -> None:
        if not self.ckpt_dir:
            return
        import torch

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self._actor.state_dict(),
                "value": self._value.state_dict(),
                "q": self._q.state_dict(),
                "succ_ema": self._succ_ema,
                "episodes": self._episodes,
            },
            self.ckpt_dir / "actor.pt",
        )

    def _maybe_log_scale(self, result) -> None:
        if not self._logged_scale and result is not None:
            a0 = np.asarray(result["actions"], dtype=np.float32)
            logger.info(
                "DB-CogACT base action stats: shape=%s min=%.3f max=%.3f std=%.3f",
                a0.shape, float(a0.min()), float(a0.max()), float(a0.std()),
            )
            self._logged_scale = True

    # ── acting: base action (single predict) + (learned) residual edit; record residual taken ──
    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import torch

        result = super().predict(obs, ctx)  # base action via inference_action; wrap sets self._last_z
        self._maybe_log_scale(result)
        if self.residual_scale <= 0 or self._last_z is None:
            return result

        z = self._last_z[:1].float().mean(dim=1)  # [1,D] (conditioned half for single predict)
        a = np.asarray(result["actions"], dtype=np.float32)  # [chunk, action_dim]
        self._ensure_rl(z.shape[-1], a.size)
        with torch.no_grad():
            residual = (self._actor(z).tanh() * self.residual_scale).cpu().numpy()[0]  # [act_flat]
            gate = 1.0
            if self.value_gate:
                gate = float((1.0 - self._value(z)).clamp(0.0, 1.0).cpu().numpy()[0, 0])

        r = residual.copy()
        sid = ctx.session_id
        if self.rl_train:
            r = r + np.random.normal(0, self.explore_std, r.shape).astype(np.float32)
            self._ep_buf.setdefault(sid, []).append(
                (z[0].detach().cpu().numpy().astype(np.float32), r.astype(np.float32))
            )
            r_exec = r * gate
        elif self.candidate_n > 0:
            r_exec = self._select_candidate(z[0], residual)
        else:
            r_exec = r * gate
        result["actions"] = a + r_exec.reshape(a.shape)
        return result

    def _select_candidate(self, z_i, r_actor):
        """Best-of-N: score {base r=0, actor, actor+noise×N} with the Q critic, pick argmax.
        Including r=0 makes this do-no-harm by construction; deviate only if Q beats base by margin."""
        import torch

        cands = [np.zeros_like(r_actor), r_actor.astype(np.float32)]
        for _ in range(self.candidate_n):
            cands.append((r_actor + np.random.normal(0, self.candidate_std, r_actor.shape)).astype(np.float32))
        cand = torch.as_tensor(np.stack(cands), device=self._rl_device)
        zr = z_i.reshape(1, -1).repeat(cand.shape[0], 1)
        with torch.no_grad():
            s = self._q(torch.cat([zr, cand], dim=1)).squeeze(1).cpu().numpy()
        best = int(np.argmax(s))
        return cands[best] if s[best] > s[0] + self.candidate_margin else cands[0]

    # ── learning: episode-level GRADED progress (completed_subtasks/5) → AWR on residuals ──
    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        buf = self._ep_buf.pop(ctx.session_id, [])
        metrics = result.get("metrics", result) if isinstance(result, dict) else {}
        progress = float(metrics.get("completed_subtasks", metrics.get("success", 0.0))) / 5.0  # graded [0,1]
        if self.rl_train and buf:
            adv_log = progress - (self._succ_ema if self._succ_ema is not None else progress)
            self._succ_ema = progress if self._succ_ema is None else (
                self.succ_ema_decay * self._succ_ema + (1 - self.succ_ema_decay) * progress
            )
            for z, r in buf:
                self._replay.append((z, r, np.float32(progress)))
            if len(self._replay) > self._replay_cap:
                self._replay = self._replay[-self._replay_cap :]
            self._episodes += 1
            loss = None
            if self._episodes >= self.warmup_episodes and len(self._replay) >= self.batch_size:
                loss = float(np.mean([self._awr_update() for _ in range(self.updates_per_episode)]))
            if self.ckpt_dir and self._episodes % self.save_every == 0:
                self._save()
            logger.info(
                "DBRLT ep=%d progress=%.2f adv=%+.3f ema=%.3f |replay|=%d awr_loss=%s",
                self._episodes, progress, adv_log, self._succ_ema or 0.0, len(self._replay),
                f"{loss:.4f}" if loss is not None else "warmup",
            )
        else:
            logger.info("DBRLT greedy progress=%.3f completed_subtasks=%.2f", progress, progress * 5.0)
        await super().on_episode_end(result, ctx)

    def _awr_update(self):
        import torch
        import torch.nn.functional as F

        batch = random.sample(self._replay, self.batch_size)
        dev = self._rl_device
        z = torch.as_tensor(np.stack([b[0] for b in batch]), device=dev)
        rtaken = torch.as_tensor(np.stack([b[1] for b in batch]), device=dev)
        ret = torch.as_tensor(np.stack([b[2] for b in batch]), device=dev).unsqueeze(1)

        v = self._value(z)
        value_loss = F.mse_loss(v, ret)
        adv = ret - v.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        w = torch.clamp(torch.exp(adv / self.awr_temp), max=self.awr_wmax)
        pred = self._actor(z).tanh() * self.residual_scale
        actor_loss = (w * (pred - rtaken).pow(2).mean(dim=1, keepdim=True)).mean() + self.l2_coef * pred.pow(2).mean()
        q_loss = F.binary_cross_entropy_with_logits(self._q(torch.cat([z, rtaken], dim=1)), ret)
        loss = actor_loss + value_loss + q_loss
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        return float(actor_loss.detach())


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(DBCogACTRLTServer)
