from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    rgb = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)
