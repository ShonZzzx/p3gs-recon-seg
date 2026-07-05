import numpy as np

from fuse_gaussian_params import fuse_gaussian_vertices


def make_vertices(xyz: np.ndarray, opacity: float) -> np.ndarray:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    out = np.zeros(len(xyz), dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["opacity"] = opacity
    out["rot_0"] = 1.0
    return out


def test_fuse_gaussian_vertices_preserves_source_rows_and_mapping() -> None:
    rade = make_vertices(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32), 1.0)
    gof = make_vertices(np.array([[0.1, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32), 2.0)

    fused, source, source_index, meta = fuse_gaussian_vertices(
        rade,
        gof,
        keep_quantile=1.0,
        fill_quantile=0.5,
        max_fill_points=10,
        seed=7,
    )

    assert fused.dtype.names == rade.dtype.names
    assert source.tolist() == ["radegs", "radegs", "gof"]
    assert source_index.tolist() == [0, 1, 1]
    assert fused["opacity"].tolist() == [1.0, 1.0, 2.0]
    assert meta["kept_rade_points"] == 2
    assert meta["added_gof_points"] == 1


def test_fuse_gaussian_vertices_can_sanitize_nonfinite_opacity() -> None:
    rade = make_vertices(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), 1.0)
    gof = make_vertices(np.array([[0.1, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32), 2.0)
    gof["opacity"][1] = np.inf

    fused, _, _, meta = fuse_gaussian_vertices(
        rade,
        gof,
        keep_quantile=1.0,
        fill_quantile=0.5,
        max_fill_points=10,
        seed=7,
        sanitize=True,
    )

    assert np.isfinite(fused["opacity"]).all()
    assert meta["nonfinite_opacity_replaced"] == 1
