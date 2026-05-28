"""Per-frame scene state extraction for RoboMME envs.

Mirrors the dataset's ``scene_state_extractor`` so a model trained on visual
trail / trajectory-text overlays can be served the same overlay at inference.
Self-contained (only ``numpy``); the model server uses ``actors`` + ``tcp_p``
to render the trail and project actor world poses to pixel coords.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class ActorRef(NamedTuple):
    attr: str
    obj: Any
    cls: str
    color: str


_NON_ACTOR_ATTRS = frozenset({"agent", "scene", "_scene", "task_list"})

_CLASS_RULES: tuple[tuple[str, str], ...] = (
    ("cube", "cube"),
    ("bin", "bin"),
    ("peg_head", "peg_head"),
    ("peg_tail", "peg_tail"),
    ("peg", "peg"),
    ("button", "button"),
    ("stick", "stick"),
    ("board", "board"),
    ("hole", "hole"),
    ("plate", "plate"),
    ("target", "target"),
)

_COLOR_NAMES: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "yellow",
    "purple",
    "orange",
    "white",
    "black",
)

_OFF_SCENE_THRESHOLD = 5.0


def _classify(attr_name: str) -> tuple[str, str]:
    name = attr_name.lower()
    cls = "unknown"
    for substr, label in _CLASS_RULES:
        if substr in name:
            cls = label
            break
    color = ""
    for c in _COLOR_NAMES:
        if c in name:
            color = c
            break
    return cls, color


def _to_xyz(p: Any) -> list[float]:
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    arr = np.asarray(p, dtype=np.float32).flatten()
    out = arr[:3].tolist()
    while len(out) < 3:
        out.append(0.0)
    return out


def _to_quat(q: Any) -> list[float]:
    if q is None:
        return [1.0, 0.0, 0.0, 0.0]
    if hasattr(q, "cpu"):
        q = q.cpu().numpy()
    arr = np.asarray(q, dtype=np.float32).flatten()
    if arr.size == 0:
        return [1.0, 0.0, 0.0, 0.0]
    out = arr[:4].tolist()
    while len(out) < 4:
        out.append(0.0)
    return out


def _is_visible(pose_p: list[float]) -> bool:
    return all(abs(c) < _OFF_SCENE_THRESHOLD for c in pose_p[:3])


def _is_actor_attr(attr: str) -> bool:
    if attr.startswith("_"):
        return False
    if attr in _NON_ACTOR_ATTRS:
        return False
    if attr in {"agent_pose", "init_pose", "robot_pose"}:
        return False
    if "joint" in attr.lower():
        return False
    return True


def _classify_score(cls: str, color: str) -> int:
    return (1 if cls != "unknown" else 0) + (1 if color else 0)


def discover_actors(env_unwrapped: Any) -> list[ActorRef]:
    """Scan env attributes for actor-like objects, deduped by ``id(obj)``.

    Two storage patterns supported: direct attribute (``self.bin_0 = actor``)
    and list/tuple attribute (``self.buttons_grid = [target0, ...]``).
    """
    best: dict[int, tuple[ActorRef, int]] = {}
    for attr in sorted(dir(env_unwrapped)):
        if not _is_actor_attr(attr):
            continue
        try:
            obj = getattr(env_unwrapped, attr, None)
        except AttributeError:
            continue
        if obj is None:
            continue

        candidates: list[tuple[str, Any]] = []
        if isinstance(obj, (list, tuple)) and not isinstance(obj, (str, bytes)):
            for i, item in enumerate(obj):
                if item is None or not hasattr(item, "pose"):
                    continue
                candidates.append((f"{attr}[{i}]", item))
        elif hasattr(obj, "pose"):
            candidates.append((attr, obj))

        for cand_attr, cand_obj in candidates:
            try:
                _ = cand_obj.pose.p
            except AttributeError:
                continue
            cls, color = _classify(cand_attr)
            actor_name = getattr(cand_obj, "name", "")
            if isinstance(actor_name, str) and actor_name:
                cls2, color2 = _classify(actor_name)
                if cls == "unknown":
                    cls = cls2
                if not color:
                    color = color2
            score = _classify_score(cls, color)
            ref = ActorRef(attr=cand_attr, obj=cand_obj, cls=cls, color=color)
            obj_id = id(cand_obj)
            prev = best.get(obj_id)
            if prev is None or score > prev[1]:
                best[obj_id] = (ref, score)
    return [ref for (ref, _) in best.values()]


def extract_scene_state(env_unwrapped: Any, actor_refs: list[ActorRef] | None = None) -> dict:
    """Per-step ``{"actors": [...], "tcp_p": [3], "tcp_q": [4]}``.

    Pass cached ``actor_refs`` from a prior ``discover_actors`` call on a hot
    path; otherwise re-discovers fresh on every call.
    """
    if actor_refs is None:
        actor_refs = discover_actors(env_unwrapped)

    actors: list[dict[str, Any]] = []
    for ref in actor_refs:
        try:
            pose = ref.obj.pose
            actor_name = getattr(ref.obj, "name", ref.attr)
            if not isinstance(actor_name, str):
                actor_name = ref.attr
            pose_p = _to_xyz(pose.p)
            actors.append(
                {
                    "name": actor_name,
                    "cls": ref.cls,
                    "color": ref.color,
                    "pose_p": pose_p,
                    "pose_q": _to_quat(getattr(pose, "q", None)),
                    "visible": _is_visible(pose_p),
                }
            )
        except AttributeError:
            continue

    tcp_p = [0.0, 0.0, 0.0]
    tcp_q = [1.0, 0.0, 0.0, 0.0]
    try:
        tcp_pose = env_unwrapped.agent.tcp_pose
        tcp_p = _to_xyz(tcp_pose.p)
        tcp_q = _to_quat(getattr(tcp_pose, "q", None))
    except Exception:
        pass

    return {"actors": actors, "tcp_p": tcp_p, "tcp_q": tcp_q}


__all__ = ["ActorRef", "discover_actors", "extract_scene_state"]
