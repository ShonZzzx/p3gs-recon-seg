from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement

from denoise_plant_gaussians import (
    build_plant_keep_mask,
    filter_full_vertices_by_plant_keep,
    write_vertices_like,
)


def make_vertices(xyz: np.ndarray) -> np.ndarray:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("rot_0", "f4"),
    ]
    vertices = np.zeros(len(xyz), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["opacity"] = np.arange(len(xyz), dtype=np.float32)
    vertices["scale_0"] = 0.1
    vertices["rot_0"] = 1.0
    return vertices


class DenoisePlantGaussiansTests(unittest.TestCase):
    def test_build_plant_keep_mask_removes_sparse_outlier_and_small_cluster(self) -> None:
        main_cluster = np.array(
            [[x * 0.01, y * 0.01, 0.0] for x in range(4) for y in range(4)],
            dtype=np.float64,
        )
        small_cluster = np.array([[1.0, 1.0, 1.0], [1.01, 1.0, 1.0]], dtype=np.float64)
        sparse_outlier = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        xyz = np.vstack([main_cluster, small_cluster, sparse_outlier])

        keep, stats = build_plant_keep_mask(
            xyz,
            sor_k=4,
            sor_mad_multiplier=3.0,
            dbscan_eps=0.05,
            dbscan_min_samples=3,
            dbscan_min_cluster_size=4,
        )

        self.assertEqual(keep[: len(main_cluster)].tolist(), [True] * len(main_cluster))
        self.assertEqual(keep[len(main_cluster) :].tolist(), [False, False, False])
        self.assertEqual(stats["plant_count"], 19)
        self.assertEqual(stats["kept_plant_count"], 16)
        self.assertEqual(stats["removed_plant_count"], 3)

    def test_filter_full_vertices_removes_only_rejected_plant_rows(self) -> None:
        full = make_vertices(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        plant_mask = np.array([False, True, True, False, True])
        plant_keep = np.array([True, False, True])

        filtered, full_keep, stats = filter_full_vertices_by_plant_keep(full, plant_mask, plant_keep)

        self.assertEqual(filtered["opacity"].tolist(), [0.0, 1.0, 3.0, 4.0])
        self.assertEqual(full_keep.tolist(), [True, True, False, True, True])
        self.assertEqual(stats["full_count"], 5)
        self.assertEqual(stats["removed_from_full_count"], 1)

    def test_write_vertices_like_preserves_property_names_and_count(self) -> None:
        vertices = make_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.ply"
            write_vertices_like(path, vertices)
            loaded = PlyData.read(path, mmap=False)["vertex"]

        self.assertEqual(loaded.count, 2)
        self.assertEqual(loaded.data.dtype.names, vertices.dtype.names)


if __name__ == "__main__":
    unittest.main()
