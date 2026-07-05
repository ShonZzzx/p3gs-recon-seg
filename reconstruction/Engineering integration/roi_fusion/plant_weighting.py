from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from plyfile import PlyData


def read_xyz_from_ply(path: str | Path) -> np.ndarray:
    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]
    required = ("x", "y", "z")
    missing = [name for name in required if name not in vertex.data.dtype.names]
    if missing:
        raise ValueError(f"{path} missing vertex properties: {', '.join(missing)}")
    return np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ],
        axis=1,
    )


def _quantized_xyz(xyz: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    return np.round(xyz / tolerance).astype(np.int64)


def build_plant_mask(
    full_ply: str | Path,
    plant_ply: str | Path,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    full_xyz = read_xyz_from_ply(full_ply)
    plant_xyz = read_xyz_from_ply(plant_ply)
    full_keys = _quantized_xyz(full_xyz, tolerance)
    plant_keys = _quantized_xyz(plant_xyz, tolerance)

    key_to_index: dict[tuple[int, int, int], int] = {}
    duplicate_full_keys = 0
    for idx, key in enumerate(map(tuple, full_keys)):
        if key in key_to_index:
            duplicate_full_keys += 1
            continue
        key_to_index[key] = idx

    mask = np.zeros(full_xyz.shape[0], dtype=np.bool_)
    unmatched = 0
    duplicate_plant_hits = 0
    for key in map(tuple, plant_keys):
        idx = key_to_index.get(key)
        if idx is None:
            unmatched += 1
            continue
        if mask[idx]:
            duplicate_plant_hits += 1
        mask[idx] = True

    meta: dict[str, float | int | str] = {
        "full_ply": str(Path(full_ply).resolve()),
        "plant_ply": str(Path(plant_ply).resolve()),
        "tolerance": tolerance,
        "full_vertices": int(full_xyz.shape[0]),
        "plant_vertices": int(plant_xyz.shape[0]),
        "matched_vertices": int(mask.sum()),
        "unmatched_vertices": int(unmatched),
        "duplicate_full_keys": int(duplicate_full_keys),
        "duplicate_plant_hits": int(duplicate_plant_hits),
    }
    return mask, meta


def save_plant_mask(mask: np.ndarray, path: str | Path, meta: dict[str, float | int | str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, plant_mask=mask.astype(np.bool_), meta=np.array([meta], dtype=object))


def load_plant_mask(path: str | Path) -> tuple[np.ndarray, dict[str, float | int | str]]:
    loaded = np.load(path, allow_pickle=True)
    mask = np.asarray(loaded["plant_mask"], dtype=np.bool_)
    meta = dict(loaded["meta"][0]) if "meta" in loaded else {}
    return mask, meta


def apply_plant_gradient_weights(
    parameters: Iterable[torch.nn.Parameter],
    plant_mask: torch.Tensor,
    plant_weight_factor: float,
) -> None:
    if plant_weight_factor <= 0:
        raise ValueError("plant_weight_factor must be > 0")
    if plant_weight_factor == 1.0:
        return
    for parameter in parameters:
        grad = parameter.grad
        if grad is None or grad.shape[0] != plant_mask.shape[0]:
            continue
        view_shape = [plant_mask.shape[0]] + [1] * (grad.dim() - 1)
        weights = torch.where(
            plant_mask.reshape(view_shape),
            torch.full((), plant_weight_factor, dtype=grad.dtype, device=grad.device),
            torch.ones((), dtype=grad.dtype, device=grad.device),
        )
        grad.mul_(weights)
