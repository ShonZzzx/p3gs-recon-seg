#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image
from plyfile import PlyData

from ply_color_utils import sh_dc_to_rgb


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def read_next_bytes(fid: BinaryIO, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as fid:
        (num_cameras,) = read_next_bytes(fid, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            _, num_params = CAMERA_MODELS[model_id]
            params = np.array(read_next_bytes(fid, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model_id, int(width), int(height), params)
    return cameras


def read_images_binary(path: Path) -> dict[int, ImageRecord]:
    images: dict[int, ImageRecord] = {}
    with path.open("rb") as fid:
        (num_images,) = read_next_bytes(fid, 8, "Q")
        for _ in range(num_images):
            binary_props = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_props[0]
            qvec = np.array(binary_props[1:5], dtype=np.float64)
            tvec = np.array(binary_props[5:8], dtype=np.float64)
            camera_id = binary_props[8]
            name_bytes = bytearray()
            while True:
                current = fid.read(1)
                if current == b"\x00":
                    break
                if current == b"":
                    raise EOFError("Unexpected end while reading image name")
                name_bytes.extend(current)
            name = name_bytes.decode("utf-8")
            (num_points2d,) = read_next_bytes(fid, 8, "Q")
            fid.seek(24 * num_points2d, 1)
            images[image_id] = ImageRecord(image_id, qvec, tvec, camera_id, name)
    return images


def find_sparse_dir(source: Path) -> Path:
    for rel in ("sparse/0", "sparse"):
        d = source / rel
        if (d / "cameras.bin").exists() and (d / "images.bin").exists():
            return d
    raise FileNotFoundError(f"No COLMAP cameras.bin/images.bin found under {source}")


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def camera_intrinsics(camera: Camera) -> tuple[float, float, float, float]:
    model, _ = CAMERA_MODELS[camera.model_id]
    p = camera.params
    if model == "SIMPLE_PINHOLE":
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    if model == "PINHOLE":
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    if model in {"SIMPLE_RADIAL", "RADIAL"}:
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    if model == "OPENCV":
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    raise ValueError(f"Camera model {model} is not supported for this lightweight projection renderer")


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.stack([np.asarray(vertex[c], dtype=np.float64) for c in ("x", "y", "z")], axis=1)
    names = set(vertex.data.dtype.names or [])
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([np.asarray(vertex[c], dtype=np.uint8) for c in ("red", "green", "blue")], axis=1)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        f_dc = np.stack([np.asarray(vertex[c], dtype=np.float64) for c in ("f_dc_0", "f_dc_1", "f_dc_2")], axis=1)
        rgb = sh_dc_to_rgb(f_dc)
    else:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    return xyz, rgb


def write_rgb_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyElement

    vertex = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"] = xyz[:, 0].astype(np.float32)
    vertex["y"] = xyz[:, 1].astype(np.float32)
    vertex["z"] = xyz[:, 2].astype(np.float32)
    vertex["red"] = rgb[:, 0].astype(np.uint8)
    vertex["green"] = rgb[:, 1].astype(np.uint8)
    vertex["blue"] = rgb[:, 2].astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def select_test_records(images: dict[int, ImageRecord], llffhold: int) -> list[ImageRecord]:
    ordered = sorted(images.values(), key=lambda item: item.name)
    return [image for idx, image in enumerate(ordered) if idx % llffhold == 0]


def select_train_records(images: dict[int, ImageRecord], llffhold: int) -> list[ImageRecord]:
    ordered = sorted(images.values(), key=lambda item: item.name)
    return [image for idx, image in enumerate(ordered) if idx % llffhold != 0]


def make_splat_offsets(radius: int) -> list[tuple[int, int]]:
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dx, dy))
    return offsets


def project_render(
    xyz: np.ndarray,
    rgb: np.ndarray,
    camera: Camera,
    image: ImageRecord,
    radius: int,
    background: int,
) -> tuple[np.ndarray, int]:
    fx, fy, cx, cy = camera_intrinsics(camera)
    r = qvec2rotmat(image.qvec)
    cam_xyz = xyz @ r.T + image.tvec.reshape(1, 3)
    z = cam_xyz[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return np.full((camera.height, camera.width, 3), background, dtype=np.uint8), 0

    cam_xyz = cam_xyz[valid]
    z = z[valid]
    colors = rgb[valid]
    u = np.rint(fx * cam_xyz[:, 0] / z + cx).astype(np.int64)
    v = np.rint(fy * cam_xyz[:, 1] / z + cy).astype(np.int64)
    valid = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    if not np.any(valid):
        return np.full((camera.height, camera.width, 3), background, dtype=np.uint8), 0

    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors = colors[valid]
    canvas = np.full((camera.height, camera.width, 3), background, dtype=np.uint8)
    zbuf = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    offsets = make_splat_offsets(radius)
    hit_count = 0
    order = np.argsort(z)[::-1]
    for dx, dy in offsets:
        uu = u[order] + dx
        vv = v[order] + dy
        zz = z[order]
        cc = colors[order]
        inside = (uu >= 0) & (uu < camera.width) & (vv >= 0) & (vv < camera.height)
        uu = uu[inside]
        vv = vv[inside]
        zz = zz[inside]
        cc = cc[inside]
        closer = zz < zbuf[vv, uu]
        if np.any(closer):
            canvas[vv[closer], uu[closer]] = cc[closer]
            zbuf[vv[closer], uu[closer]] = zz[closer]
            hit_count += int(np.count_nonzero(closer))
    return canvas, hit_count


def copy_gt(source: Path, image_name: str, out_path: Path) -> bool:
    src = source / "images" / image_name
    if not src.exists():
        src = source / image_name
    if not src.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(out_path)
    return True


def image_path_for_record(source: Path, record: ImageRecord) -> Path | None:
    for candidate in (source / "images" / record.name, source / record.name):
        if candidate.exists():
            return candidate
    return None


def colorize_from_train_views(
    xyz: np.ndarray,
    fallback_rgb: np.ndarray,
    source: Path,
    cameras: dict[int, Camera],
    records: list[ImageRecord],
    max_views: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    if max_views > 0:
        records = records[:max_views]
    sums = np.zeros((len(xyz), 3), dtype=np.float64)
    counts = np.zeros(len(xyz), dtype=np.uint16)
    used = []
    for record in records:
        img_path = image_path_for_record(source, record)
        if img_path is None:
            continue
        camera = cameras[record.camera_id]
        fx, fy, cx, cy = camera_intrinsics(camera)
        r = qvec2rotmat(record.qvec)
        with Image.open(img_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        for start in range(0, len(xyz), batch_size):
            end = min(start + batch_size, len(xyz))
            cam_xyz = xyz[start:end] @ r.T + record.tvec.reshape(1, 3)
            z = cam_xyz[:, 2]
            valid = z > 1e-6
            if not np.any(valid):
                continue
            local_idx = np.nonzero(valid)[0]
            pts = cam_xyz[valid]
            zz = z[valid]
            u = np.rint(fx * pts[:, 0] / zz + cx).astype(np.int64)
            v = np.rint(fy * pts[:, 1] / zz + cy).astype(np.int64)
            inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
            if not np.any(inside):
                continue
            global_idx = start + local_idx[inside]
            samples = arr[v[inside], u[inside]].astype(np.float64)
            sums[global_idx] += samples
            counts[global_idx] += 1
        used.append(record.name)
    out = fallback_rgb.copy()
    colored = counts > 0
    out[colored] = np.rint(sums[colored] / counts[colored, None]).clip(0, 255).astype(np.uint8)
    return out, {"used_train_views": used, "colored_points": int(np.count_nonzero(colored)), "total_points": int(len(xyz))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="COLMAP dataset root, e.g. dataGS/plant_002_undistorted")
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--background", type=int, default=255)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--colorize-from-train", action="store_true")
    parser.add_argument("--colorized-ply", type=Path, default=None)
    parser.add_argument("--max-color-views", type=int, default=0)
    parser.add_argument("--color-batch-size", type=int, default=200000)
    args = parser.parse_args()

    sparse = find_sparse_dir(args.source)
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    records = select_test_records(images, args.llffhold)
    if args.max_images > 0:
        records = records[: args.max_images]

    xyz, rgb = read_ply(args.ply)
    colorize_meta = None
    if args.colorize_from_train:
        rgb, colorize_meta = colorize_from_train_views(
            xyz,
            rgb,
            args.source,
            cameras,
            select_train_records(images, args.llffhold),
            args.max_color_views,
            args.color_batch_size,
        )
        if args.colorized_ply is not None:
            write_rgb_ply(args.colorized_ply, xyz, rgb)
    render_dir = args.out_dir / "renders" / "test"
    gt_dir = args.out_dir / "gt" / "test"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in records:
        camera = cameras[record.camera_id]
        render, hits = project_render(xyz, rgb, camera, record, args.splat_radius, args.background)
        out_name = Path(record.name).with_suffix(".png").name
        Image.fromarray(render).save(render_dir / out_name)
        gt_ok = copy_gt(args.source, record.name, gt_dir / out_name)
        rows.append({"image": record.name, "output": out_name, "projected_pixels": hits, "gt_copied": gt_ok})

    meta = {
        "source": str(args.source),
        "sparse": str(sparse),
        "ply": str(args.ply),
        "out_dir": str(args.out_dir),
        "llffhold": args.llffhold,
        "splat_radius": args.splat_radius,
        "background": args.background,
        "num_images": len(rows),
        "rows": rows,
        "render_type": "ply_projection_zbuffer",
        "colorize_from_train": bool(args.colorize_from_train),
        "colorize_meta": colorize_meta,
    }
    (args.out_dir / "render_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {args.out_dir}")
    print(f"IMAGES {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
