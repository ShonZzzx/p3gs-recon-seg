import numpy as np

from render_ply_colmap_cameras import Camera, ImageRecord, camera_intrinsics, select_test_records, select_train_records


def make_image(name: str) -> ImageRecord:
    return ImageRecord(
        image_id=int(name.split(".")[0]),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        camera_id=1,
        name=name,
    )


def test_records_use_sorted_llff_holdout() -> None:
    images = {i: make_image(f"{i:03d}.png") for i in range(10)}

    selected = select_test_records(images, llffhold=4)

    assert [item.name for item in selected] == ["000.png", "004.png", "008.png"]


def test_train_records_are_holdout_complement() -> None:
    images = {i: make_image(f"{i:03d}.png") for i in range(10)}

    selected = select_train_records(images, llffhold=4)

    assert [item.name for item in selected] == ["001.png", "002.png", "003.png", "005.png", "006.png", "007.png", "009.png"]


def test_camera_intrinsics_support_pinhole_models() -> None:
    simple = Camera(1, 0, 800, 600, np.array([500.0, 400.0, 300.0]))
    pinhole = Camera(2, 1, 800, 600, np.array([510.0, 520.0, 401.0, 302.0]))

    assert camera_intrinsics(simple) == (500.0, 500.0, 400.0, 300.0)
    assert camera_intrinsics(pinhole) == (510.0, 520.0, 401.0, 302.0)
