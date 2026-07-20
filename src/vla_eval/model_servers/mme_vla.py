# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openpi",
#     "numpy>=1.24",
#     "huggingface_hub",
#     "pytest",  # not declared in openpi's deps but imported by openpi.models_pytorch
#     "chex",    # not declared in openpi's deps but imported by openpi.models
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# openpi = { git = "https://github.com/RoboMME/robomme_policy_learning.git", rev = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b" }
#
# [tool.uv]
# exclude-newer = "2026-04-06T00:00:00Z"
# ///
"""MME-VLA model server for RoboMME baselines.

Loads MME-VLA suite checkpoints (pi0.5 baseline + memory-augmented
variants) and runs inference in-process.  Supports the ``video_history``
conditioning protocol used by the RoboMME benchmark.

Memory-augmented models (FrameSamp, TokenDrop, TTT, RMT) carry a streaming
memory of the episode. Their ``infer`` asserts on an empty buffer, so with
``use_history=true`` every observation is staged and the batch since the last
inference is handed to ``add_buffer`` right before each ``infer`` — mirroring the
upstream rollout's ``add_observation`` / ``add_buffer`` + ``clear_buffers`` pair.
A conditioning demo clip, for the tasks that have one, is merely the prefix of
the first batch and sets ``exec_start_idx``; a task without one (``exec_start_idx``
0 throughout) still works and simply remembers only its own history. The baseline
pi0.5 model has no memory and ignores all of this.

``use_history`` must match the checkpoint, not the task: the assertion lives in
the policy, so a memory variant raises on every observation when it is served
with ``use_history=false``, demo clip or not.

The ``openpi`` PEP 723 dependency installs the RoboMME fork of OpenPI,
which includes both the ``openpi`` and ``mme_vla_suite`` Python modules.
Two configs are registered: ``pi05_baseline`` (no memory) and
``mme_vla_suite`` (all 14 memory-augmented variants — the specific
architecture is determined by the checkpoint's saved config).
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

# The RoboMME fork of OpenPI ships both ``openpi`` and ``mme_vla_suite`` under ``src/``, but
# hatchling only builds the ``openpi`` wheel.  Shallow-clone the repo at runtime so
# ``mme_vla_suite`` is importable.
_MME_VLA_REPO = "https://github.com/RoboMME/robomme_policy_learning.git"
_MME_VLA_REV = "main"


def _ensure_mme_vla_suite() -> None:
    """Make the ``mme_vla_suite`` package importable."""
    try:
        import mme_vla_suite  # noqa: F401

        return
    except ImportError:
        pass

    from vla_eval.dirs import ensure_git_clone

    clone = ensure_git_clone(name="mme-vla", repo=_MME_VLA_REPO, rev=_MME_VLA_REV, shallow=True)
    src_dir = str(clone / "src")
    # Append (not insert) so the installed openpi wheel still takes priority
    sys.path.append(src_dir)
    import mme_vla_suite  # noqa: F401, F811

    logger.info("mme_vla_suite loaded from %s", src_dir)


class MmeVlaModelServer(PredictModelServer):
    """MME-VLA suite model server for RoboMME evaluation.

    Handles both the pi0.5 baseline (no memory) and all 14 memory-augmented variants from the
    MME-VLA paper.

    Args:
        config_name: MME-VLA config — ``"pi05_baseline"`` or ``"mme_vla_suite"`` (memory variants).
        checkpoint: HuggingFace model ID or local path.  For the multi-variant repo, use
            ``Yinpei/mme_vla_suite/subdir``.
        use_history: Enable the memory lifecycle (reset + streaming add_buffer). This is a
            property of the CHECKPOINT, not the task: memory-augmented variants assert on an
            empty buffer inside ``infer``, so they require ``True`` even for tasks with no
            conditioning demo. Leave ``False`` only for the pi0.5 baseline.
        image_key: Key for the front camera in the OpenPI obs dict.
        wrist_image_key: Key for the wrist camera (``None`` to disable).
        state_key: Key for proprioceptive state (``None`` to disable).
        state_dim: Truncate benchmark state to this dimension.  RoboMME sends 9D; models expect 8D.
        image_resolution: Resize images to this square resolution.
        chunk_size: Number of actions per inference call.
        action_ensemble: Ensemble strategy for overlapping chunks.
    """

    def __init__(
        self,
        config_name: str = "pi05_baseline",
        checkpoint: str | None = None,
        use_history: bool = False,
        image_key: str = "observation/image",
        wrist_image_key: str | None = "observation/wrist_image",
        state_key: str | None = "observation/state",
        state_dim: int = 8,
        image_resolution: int | None = 224,
        *,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        if use_history and kwargs.get("max_batch_size", 1) > 1:
            raise ValueError("use_history=True is incompatible with max_batch_size > 1 (memory is per-session)")
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.use_history = use_history
        self.image_key = image_key
        self.wrist_image_key = None if wrist_image_key in (None, "None", "none") else wrist_image_key
        self.state_key = None if state_key in (None, "None", "none") else state_key
        self.state_dim = state_dim
        self.image_resolution = image_resolution
        self._state_warned = False
        # Per-session frames staged since the last inference (see _stage_frame).
        self._pending: dict[str, dict[str, Any]] = {}

        _ensure_mme_vla_suite()

        from mme_vla_suite.policies import policy_config
        from mme_vla_suite.training import config as _config

        logger.info("Loading MME-VLA config: %s", self.config_name)
        _cfg = _config.get_config(self.config_name)
        ckpt_path = pathlib.Path(self._resolve_checkpoint())
        logger.info("Loading policy from checkpoint: %s", ckpt_path)
        self._policy = policy_config.create_trained_policy(_cfg, ckpt_path)
        logger.info("MME-VLA policy loaded successfully.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_checkpoint(self) -> str:
        """Resolve checkpoint to a local directory containing ``params/`` and ``assets/``."""
        if self.checkpoint is None:
            raise ValueError("checkpoint must be specified for MME-VLA models")

        # Already a local path with params/
        if os.path.isdir(self.checkpoint) and os.path.isdir(os.path.join(self.checkpoint, "params")):
            return self.checkpoint

        from vla_eval.dirs import require_model_available

        require_model_available(self.checkpoint)

        from huggingface_hub import snapshot_download

        # Handle 3-segment paths: org/repo/subdir
        parts = self.checkpoint.split("/")
        if len(parts) >= 3:
            repo_id = "/".join(parts[:2])
            subdir = "/".join(parts[2:])
            local = snapshot_download(repo_id)
            dl_path = os.path.join(local, subdir)
        else:
            dl_path = snapshot_download(self.checkpoint)

        return self._find_or_extract_checkpoint(dl_path)

    @staticmethod
    def _find_or_extract_checkpoint(dl_path: str) -> str:
        """Find checkpoint root (containing ``params/``) inside *dl_path*.

        HuggingFace repos may ship checkpoints as zip files with an
        internal directory tree.  This method extracts the zip if needed
        and walks the result to locate the ``params/`` directory.
        """
        # Direct match
        if os.path.isdir(os.path.join(dl_path, "params")):
            return dl_path

        # Look for zip files to extract
        import glob
        import zipfile

        zips = glob.glob(os.path.join(dl_path, "**/*.zip"), recursive=True)
        if not zips:
            zips = glob.glob(os.path.join(dl_path, "*.zip"))

        extract_base = os.path.join(dl_path, "_extracted")
        for zf in zips:
            extract_dir = os.path.join(extract_base, pathlib.Path(zf).stem)
            if not os.path.isdir(extract_dir):
                logger.info("Extracting checkpoint: %s", zf)
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(extract_dir)

        # Walk to find the directory that contains params/
        search_root = extract_base if os.path.isdir(extract_base) else dl_path
        for root, dirs, _files in os.walk(search_root):
            if "params" in dirs:
                logger.info("Found checkpoint root: %s", root)
                return root

        raise FileNotFoundError(f"No params/ directory found in {dl_path}")

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _maybe_resize(self, img: np.ndarray) -> np.ndarray:
        if self.image_resolution is None or img.shape[:2] == (self.image_resolution, self.image_resolution):
            return img
        from PIL import Image

        pil = Image.fromarray(img)
        pil = pil.resize((self.image_resolution, self.image_resolution), Image.Resampling.BILINEAR)
        return np.asarray(pil)

    # ------------------------------------------------------------------
    # Video history → add_buffer
    # ------------------------------------------------------------------

    def _stage_frame(self, obs: Observation, ctx: SessionContext) -> None:
        """Accumulate one observation into the pending memory batch.

        Mirrors ``EpisodeState.add_observation`` in the upstream rollout: every
        observation is staged, and the batch accumulated since the last
        inference is handed to ``add_buffer`` when inference next runs. Only the
        base camera enters memory (the buffer carries ``num_views=1``); the wrist
        image is a per-step input, not a remembered one.
        """
        images_dict = obs.get("images", {})
        img_list = list(images_dict.values()) if isinstance(images_dict, dict) else []
        if not img_list:
            return
        pend = self._pending.setdefault(ctx.session_id, {"images": [], "states": [], "exec_start_idx": 0})

        # A demo clip, when the task has one, is the prefix of the very first
        # batch — exactly as upstream seeds image_buffer from get_init_obs()
        # before appending the first live frame.
        if ctx.is_first:
            demo = obs.get("video_history") or []
            if demo:
                pend["images"].extend(self._maybe_resize(np.asarray(f, dtype=np.uint8)) for f in demo)
                pend["states"].extend(np.zeros(self.state_dim, dtype=np.float32) for _ in demo)
                pend["exec_start_idx"] = len(demo)

        pend["images"].append(self._maybe_resize(np.asarray(img_list[0], dtype=np.uint8)))
        pend["states"].append(self._obs_state(obs).astype(np.float32))

    def _flush_buffer(self, ctx: SessionContext) -> None:
        """Hand the staged batch to the policy's memory, then clear it.

        ``MME_VLA_Policy.add_buffer`` expects:
            images: ``(T, 1, H, W, 3)`` uint8 — the extra axis is ``num_views``
            state:  ``(T, state_dim)`` float32
            exec_start_idx: int — index of the first execution frame

        Called immediately before every inference, and the staged batch is
        incremental: the policy advances its own ``step_idx`` by ``len(images)``,
        so re-sending frames it already holds would corrupt the timeline. This is
        upstream's ``add_buffer(...)`` + ``clear_buffers()`` pair.

        ``exec_start_idx`` is only meaningful on the first batch — the policy
        keeps its own copy and ignores 0 on later calls, so a task with no demo
        (exec_start_idx 0 throughout) simply remembers its own history. That is
        the normal case here: the real-robot task1 data was converted with
        ``is_video_demo=False`` on every frame, so the model was trained with a
        buffer holding nothing but the episode's own past.
        """
        pend = self._pending.get(ctx.session_id)
        if not pend or not pend["images"]:
            return
        images = np.stack(pend["images"])[:, np.newaxis]  # (T, 1, H, W, 3)
        buffer_obs = {
            "images": images,
            "state": np.stack(pend["states"]),
            "exec_start_idx": pend["exec_start_idx"],
        }
        assert self._policy is not None
        self._policy.add_buffer(buffer_obs)
        logger.debug(
            "add_buffer: %d frame(s), exec_start_idx=%d, session=%s",
            len(pend["images"]),
            pend["exec_start_idx"],
            ctx.session_id[:8],
        )
        self._pending[ctx.session_id] = {"images": [], "states": [], "exec_start_idx": 0}

    def _obs_state(self, obs: Observation) -> np.ndarray:
        """Proprioceptive state, truncated to the model's width."""
        raw_state = obs.get("states", obs.get("state"))
        if raw_state is None:
            return np.zeros(self.state_dim, dtype=np.float64)
        state = np.asarray(raw_state, dtype=np.float64)
        if state.shape[0] > self.state_dim:
            if not self._state_warned:
                logger.info("Truncating state from %dD to %dD", state.shape[0], self.state_dim)
                self._state_warned = True
            state = state[: self.state_dim]
        return state

    # ------------------------------------------------------------------
    # Specs and params
    # ------------------------------------------------------------------

    def get_observation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.wrist_image_key:
            params["send_wrist_image"] = True
        if self.state_key:
            params["send_state"] = True
        if self.use_history:
            params["send_video_history"] = True
        return params

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB}
        if self.state_key:
            spec["state"] = RAW
        spec["language"] = LANGUAGE
        return spec

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_start(config, ctx)
        self._pending.pop(ctx.session_id, None)
        if self.use_history and self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        """Stage every observation before the base class decides whether to infer.

        Memory must see the episode at full rate. Inference does not run on every
        observation — the base class serves most of them from the action chunk
        buffer and only calls :meth:`predict` once the chunk is exhausted — so
        staging here rather than in ``predict`` is what keeps the remembered
        history continuous instead of subsampled to one frame per chunk. The
        dataset builder adds one frame per step; this matches it.
        """
        if self.use_history:
            self._stage_frame(obs, ctx)
        await super().on_observation(obs, ctx)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        # Memory variants assert on an empty buffer inside infer(), so the staged
        # frames must land before it — upstream pairs add_buffer with every
        # inference, not just the first.
        if self.use_history:
            self._flush_buffer(ctx)

        # Build OpenPI observation dict
        openpi_obs: dict[str, Any] = {}

        # Images
        images_dict = obs.get("images", {})
        img_list = list(images_dict.values()) if isinstance(images_dict, dict) else []
        base_img = np.asarray(img_list[0], dtype=np.uint8) if img_list else np.zeros((256, 256, 3), dtype=np.uint8)
        base_img = self._maybe_resize(base_img)
        openpi_obs[self.image_key] = base_img

        if self.wrist_image_key:
            wrist_img = np.asarray(img_list[1], dtype=np.uint8) if len(img_list) > 1 else np.zeros_like(base_img)
            wrist_img = self._maybe_resize(wrist_img)
            openpi_obs[self.wrist_image_key] = wrist_img

        # Prompt (lowercase to match upstream convention)
        openpi_obs["prompt"] = obs.get("task_description", "").lower()

        # State (truncate to model's expected dimension)
        if self.state_key:
            openpi_obs[self.state_key] = self._obs_state(obs)

        result = self._policy.infer(openpi_obs)
        return {"actions": result["actions"]}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(MmeVlaModelServer)
