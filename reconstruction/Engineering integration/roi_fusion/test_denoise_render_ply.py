from pathlib import Path

from denoise_render_ply import find_input_ply


def test_find_input_ply_uses_largest_numeric_iteration(tmp_path: Path) -> None:
    root = tmp_path
    for iteration in ("iteration_7000", "iteration_30000"):
        out = root / "gof" / "plant_002" / "point_cloud" / iteration
        out.mkdir(parents=True)
        (out / "point_cloud.ply").write_text("ply\n", encoding="utf-8")

    selected = find_input_ply(root, "gof", "plant_002")

    assert selected == root / "gof" / "plant_002" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
