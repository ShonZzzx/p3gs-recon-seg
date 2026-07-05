from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from plant_weighting import apply_plant_gradient_weights, build_plant_mask


def write_ply(path: Path, vertices: np.ndarray) -> None:
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def make_vertices(xyz: np.ndarray, order: str = "full") -> np.ndarray:
    if order == "cloudcompare":
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("scale_0", "f4"),
            ("f_dc_0", "f4"),
            ("opacity", "f4"),
            ("rot_0", "f4"),
        ]
    else:
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("f_dc_0", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("rot_0", "f4"),
        ]
    vertices = np.zeros(len(xyz), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["opacity"] = 1.0
    vertices["rot_0"] = 1.0
    return vertices


class PlantWeightingTests(unittest.TestCase):
    def test_build_plant_mask_matches_subset_by_xyz_even_when_property_order_differs(self) -> None:
        full_xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        plant_xyz = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            full_ply = Path(tmpdir) / "full.ply"
            plant_ply = Path(tmpdir) / "plant.ply"
            write_ply(full_ply, make_vertices(full_xyz, "full"))
            write_ply(plant_ply, make_vertices(plant_xyz, "cloudcompare"))

            mask, meta = build_plant_mask(full_ply, plant_ply, tolerance=1e-6)

        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.tolist(), [True, False, True, False])
        self.assertEqual(meta["full_vertices"], 4)
        self.assertEqual(meta["plant_vertices"], 2)
        self.assertEqual(meta["matched_vertices"], 2)
        self.assertEqual(meta["unmatched_vertices"], 0)

    def test_apply_plant_gradient_weights_scales_first_dimension_rows_only(self) -> None:
        plant_mask = torch.tensor([True, False, True])
        xyz = torch.nn.Parameter(torch.ones(3, 3))
        features = torch.nn.Parameter(torch.ones(3, 1, 2))
        opacity = torch.nn.Parameter(torch.ones(3, 1))
        other = torch.nn.Parameter(torch.ones(2, 1))
        xyz.grad = torch.ones_like(xyz)
        features.grad = torch.ones_like(features)
        opacity.grad = torch.ones_like(opacity)
        other.grad = torch.ones_like(other)

        apply_plant_gradient_weights([xyz, features, opacity, other], plant_mask, plant_weight_factor=3.0)

        self.assertEqual(xyz.grad[:, 0].tolist(), [3.0, 1.0, 3.0])
        self.assertEqual(features.grad[:, 0, 0].tolist(), [3.0, 1.0, 3.0])
        self.assertEqual(opacity.grad[:, 0].tolist(), [3.0, 1.0, 3.0])
        self.assertEqual(other.grad[:, 0].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
