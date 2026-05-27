"""LeRobotDataset v3 ingestion (parquet), lazy and torch-free.

The loaders read a v3 dataset's ``meta/info.json`` (fps, robot_type, joint names) and the
episode ``parquet`` shards directly with ``pyarrow`` — the ``lerobot`` library and ``torch`` are
never imported. ``pyarrow``/``huggingface_hub`` are imported lazily (the ``hub`` extra), so the
in-memory :func:`tracecal.io.lerobot_v3.from_arrays` path stays usable with the core install.
"""

from __future__ import annotations
