import numpy as np

from ply_color_utils import sh_dc_to_rgb


def test_sh_dc_zero_maps_to_mid_gray() -> None:
    f_dc = np.zeros((2, 3), dtype=np.float64)

    rgb = sh_dc_to_rgb(f_dc)

    assert rgb.dtype == np.uint8
    assert rgb.tolist() == [[128, 128, 128], [128, 128, 128]]


def test_sh_dc_values_are_clamped_to_rgb_range() -> None:
    f_dc = np.array([[-10.0, 0.0, 10.0]], dtype=np.float64)

    rgb = sh_dc_to_rgb(f_dc)

    assert rgb.tolist() == [[0, 128, 255]]
