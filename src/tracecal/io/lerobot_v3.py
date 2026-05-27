"""Load LeRobotDataset v3 episodes into the tracecal IR.

A v3 dataset on disk looks like::

    <root>/meta/info.json           # fps, robot_type, features (with per-feature `names`)
    <root>/data/chunk-*/*.parquet   # frames; columns include episode_index, frame_index,
                                    # observation.state (joint positions), optionally success

This module groups frames by ``episode_index``, orders them by ``frame_index`` and stacks the
chosen state column into a ``(T, dof)`` joint-position array per episode. ``pyarrow`` is imported
lazily; the in-memory :func:`from_arrays` path needs none of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tracecal.io.labels import SUCCESS_KEYS, detect_episode_label
from tracecal.schema import EpisodeRecord

DEFAULT_STATE_KEY = "observation.state"


@dataclass(frozen=True)
class LoadedEpisode:
    """One episode's metadata record plus its (T, dof) joint-position trajectory."""

    record: EpisodeRecord
    positions: np.ndarray


@dataclass(frozen=True)
class LeRobotV3Dataset:
    """A loaded (or in-memory) LeRobot v3 dataset ready for the audit pipeline."""

    robot_type: str
    fps: float
    joint_names: tuple[str, ...]
    episodes: tuple[LoadedEpisode, ...]
    source: str

    def __len__(self) -> int:
        return len(self.episodes)


def from_arrays(
    *,
    robot_type: str,
    fps: float,
    joint_names: tuple[str, ...],
    episodes: list[tuple[str, np.ndarray]] | list[tuple[str, np.ndarray, float | None]],
    dataset: str | None = None,
    source: str = "in-memory",
) -> LeRobotV3Dataset:
    """Build a dataset from in-memory ``(episode_id, positions[, success])`` tuples (no pyarrow).

    This is the offline path used by tests and by callers that already have trajectories.
    """
    dof = len(joint_names)
    loaded: list[LoadedEpisode] = []
    for ep in episodes:
        ep_id, positions = ep[0], np.asarray(ep[1], dtype=float)
        success = ep[2] if len(ep) > 2 else None
        if positions.ndim != 2 or positions.shape[1] != dof:
            raise ValueError(
                f"episode {ep_id!r}: positions must be (T, {dof}); got {positions.shape}."
            )
        rec = EpisodeRecord(
            id=str(ep_id),
            embodiment_id=robot_type,
            fps=fps,
            n_steps=int(positions.shape[0]),
            joint_names=tuple(joint_names),
            dataset=dataset,
            success=success,
        )
        loaded.append(LoadedEpisode(record=rec, positions=positions))
    return LeRobotV3Dataset(
        robot_type=robot_type,
        fps=float(fps),
        joint_names=tuple(joint_names),
        episodes=tuple(loaded),
        source=source,
    )


def _read_info(root: Path, state_key: str) -> tuple[float, str, tuple[str, ...]]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"no meta/info.json under {root} (is this a LeRobot v3 dataset?)")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 0.0))
    if fps <= 0.0:
        raise ValueError("info.json has no positive 'fps'.")
    robot_type = str(info.get("robot_type") or info.get("robot") or "unknown")
    feats = info.get("features", {})
    if state_key not in feats:
        raise KeyError(f"state feature {state_key!r} not in info.json features {list(feats)}.")
    names = feats[state_key].get("names")
    joint_names = _flatten_names(names)
    return fps, robot_type, joint_names


def _flatten_names(names: Any) -> tuple[str, ...]:
    """Normalise the v3 `names` field (list, or dict-of-lists like {'motors': [...]})."""
    if names is None:
        return ()
    if isinstance(names, dict):
        out: list[str] = []
        for v in names.values():
            out.extend(str(x) for x in v)
        return tuple(out)
    return tuple(str(x) for x in names)


def load_local(
    root: str | Path,
    *,
    state_key: str = DEFAULT_STATE_KEY,
    max_episodes: int | None = None,
) -> LeRobotV3Dataset:
    """Load a v3 dataset from a local directory using pyarrow (the ``hub`` extra)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the hub extra
        raise ImportError(
            "loading LeRobot datasets needs the 'hub' extra: pip install 'tracecal[hub]'."
        ) from exc

    root = Path(root)
    fps, robot_type, joint_names = _read_info(root, state_key)
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no data/**/*.parquet under {root}.")

    # Accumulate per-episode (frame_index, state-row) then stack.
    by_ep: dict[int, list[tuple[int, np.ndarray]]] = {}
    success_cols: dict[int, list[Any]] = {}
    label_source: str | None = None
    for pf in parquet_files:
        # pyarrow ships py.typed but read_table is effectively untyped; route through Any so
        # strict mypy is satisfied whether or not pyarrow is installed in the checking env.
        read_table: Any = pq.read_table
        table = read_table(pf)
        cols = table.column_names
        if state_key not in cols or "episode_index" not in cols:
            continue
        ep_idx = table.column("episode_index").to_pylist()
        fr_idx = (
            table.column("frame_index").to_pylist()
            if "frame_index" in cols
            else list(range(len(ep_idx)))
        )
        states = table.column(state_key).to_pylist()
        succ_key = next((k for k in SUCCESS_KEYS if k in cols), None)
        succ_vals = table.column(succ_key).to_pylist() if succ_key else None
        if succ_key:
            label_source = succ_key
        for i, e in enumerate(ep_idx):
            by_ep.setdefault(int(e), []).append(
                (int(fr_idx[i]), np.asarray(states[i], dtype=float))
            )
            if succ_vals is not None:
                success_cols.setdefault(int(e), []).append(succ_vals[i])

    episode_ids = sorted(by_ep)
    if max_episodes is not None:
        episode_ids = episode_ids[:max_episodes]

    out: list[tuple[str, np.ndarray, float | None]] = []
    for e in episode_ids:
        frames = sorted(by_ep[e], key=lambda t: t[0])
        positions = np.stack([row for _, row in frames], axis=0)
        label, _ = detect_episode_label(
            {label_source: np.asarray(success_cols.get(e, []))} if label_source else {},
        )
        out.append((f"episode_{e:06d}", positions, label))

    return from_arrays(
        robot_type=robot_type,
        fps=fps,
        joint_names=joint_names,
        episodes=out,
        dataset=str(root.name),
        source=f"local:{root}",
    )


def load_hub(
    repo_id: str,
    *,
    revision: str = "main",
    state_key: str = DEFAULT_STATE_KEY,
    max_episodes: int | None = None,
) -> LeRobotV3Dataset:
    """Download a public dataset from the Hugging Face Hub, then load it (the ``hub`` extra)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "loading from the Hub needs the 'hub' extra: pip install 'tracecal[hub]'."
        ) from exc

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["meta/*", "data/**"],
    )
    ds = load_local(local, state_key=state_key, max_episodes=max_episodes)
    return LeRobotV3Dataset(
        robot_type=ds.robot_type,
        fps=ds.fps,
        joint_names=ds.joint_names,
        episodes=ds.episodes,
        source=f"hub:{repo_id}@{revision}",
    )
