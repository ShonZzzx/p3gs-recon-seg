#!/usr/bin/env python3
"""Create a plant-centered ROI COLMAP dataset without rerunning COLMAP.

The crop is driven by registered COLMAP 2D points for each image. This keeps
the original full-image COLMAP poses, crops images around the reconstructed
object evidence, and writes adjusted camera intrinsics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageDraw


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3, (1, 2)),
    1: ("PINHOLE", 4, (2, 3)),
    2: ("SIMPLE_RADIAL", 4, (1, 2)),
    3: ("RADIAL", 5, (1, 2)),
    4: ("OPENCV", 8, (2, 3)),
    5: ("OPENCV_FISHEYE", 8, (2, 3)),
    6: ("FULL_OPENCV", 12, (2, 3)),
    7: ("FOV", 5, (2, 3)),
    8: ("SIMPLE_RADIAL_FISHEYE", 4, (1, 2)),
    9: ("RADIAL_FISHEYE", 5, (1, 2)),
    10: ("THIN_PRISM_FISHEYE", 12, (2, 3)),
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
    xys: np.ndarray
    point3d_ids: np.ndarray


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
            model_name, num_params, _ = CAMERA_MODELS.get(model_id, (None, None, None))
            if model_name is None:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            params = np.array(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = Camera(camera_id, model_id, width, height, params)
    return cameras


def write_cameras_binary(cameras: dict[int, Camera], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras.values():
            fid.write(struct.pack("<iiQQ", camera.camera_id, camera.model_id, camera.width, camera.height))
            fid.write(struct.pack("<" + "d" * len(camera.params), *camera.params.tolist()))


def read_images_binary(path: Path) -> dict[int, ImageRecord]:
    images: dict[int, ImageRecord] = {}
    with path.open("rb") as fid:
        (num_images,) = read_next_bytes(fid, 8, "Q")
        for _ in range(num_images):
            binary_props = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_props[0]
            qvec = np.array(binary_props[1:5])
            tvec = np.array(binary_props[5:8])
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
            xys = np.empty((num_points2d, 2), dtype=np.float64)
            point3d_ids = np.empty(num_points2d, dtype=np.int64)
            for i in range(num_points2d):
                x, y, pid = read_next_bytes(fid, 24, "ddq")
                xys[i] = (x, y)
                point3d_ids[i] = pid
            images[image_id] = ImageRecord(image_id, qvec, tvec, camera_id, name, xys, point3d_ids)
    return images


def write_images_binary(images: dict[int, ImageRecord], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images.values():
            fid.write(
                struct.pack(
                    "<idddddddi",
                    image.image_id,
                    *image.qvec.tolist(),
                    *image.tvec.tolist(),
                    image.camera_id,
                )
            )
            fid.write(image.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image.xys)))
            for (x, y), pid in zip(image.xys, image.point3d_ids):
                fid.write(struct.pack("<ddq", float(x), float(y), int(pid)))


def find_sparse_dir(source: Path) -> Path:
    for rel in ("sparse/0", "sparse"):
        d = source / rel
        if (d / "cameras.bin").exists() and (d / "images.bin").exists() and (d / "points3D.bin").exists():
            return d
    raise FileNotFoundError(f"No COLMAP sparse binary model found under {source}")


def crop_from_points(
    xys: np.ndarray,
    width: int,
    height: int,
    percentile: float,
    margin: float,
    min_points: int,
    min_size: int,
) -> tuple[int, int, int, int]:
    valid = xys[np.isfinite(xys).all(axis=1)]
    if len(valid) < min_points:
        raise ValueError(f"Only {len(valid)} valid COLMAP points, need at least {min_points}")
    lo = percentile
    hi = 100.0 - percentile
    x0, y0 = np.percentile(valid, [lo], axis=0)[0]
    x1, y1 = np.percentile(valid, [hi], axis=0)[0]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    x0 -= bw * margin
    x1 += bw * margin
    y0 -= bh * margin
    y1 += bh * margin
    if x1 - x0 < min_size:
        pad = (min_size - (x1 - x0)) / 2
        x0 -= pad
        x1 += pad
    if y1 - y0 < min_size:
        pad = (min_size - (y1 - y0)) / 2
        y0 -= pad
        y1 += pad
    left = max(0, int(np.floor(x0)))
    upper = max(0, int(np.floor(y0)))
    right = min(width, int(np.ceil(x1)))
    lower = min(height, int(np.ceil(y1)))
    if right <= left or lower <= upper:
        raise ValueError("Empty crop")
    return left, upper, right, lower


def crop_from_green_foreground(
    image: Image.Image,
    margin: float,
    min_size: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
) -> tuple[int, int, int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for green foreground ROI mode") from exc

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    exg = 2 * g - r - b

    green_hue = (hue >= 35) & (hue <= 98) & (sat >= 35) & (val >= 30)
    excess_green = (exg >= 18) & (g > r + 8) & (g > b + 8) & (sat >= 25)
    mask = (green_hue | excess_green).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        raise ValueError("No foreground component found")

    areas = stats[1:, cv2.CC_STAT_AREA]
    total_area = float(h * w)
    keep = []
    min_component_area = max(40, int(total_area * 0.0004))
    for idx, area in enumerate(areas, start=1):
        if area >= min_component_area:
            keep.append(idx)
    if not keep:
        raise ValueError("Foreground components are too small")

    kept = np.isin(labels, keep)
    ratio = float(kept.sum()) / total_area
    if ratio < min_mask_ratio or ratio > max_mask_ratio:
        raise ValueError(f"Foreground mask ratio {ratio:.4f} outside [{min_mask_ratio}, {max_mask_ratio}]")

    ys, xs = np.where(kept)
    left, right = int(xs.min()), int(xs.max()) + 1
    upper, lower = int(ys.min()), int(ys.max()) + 1
    bw = max(1, right - left)
    bh = max(1, lower - upper)
    left = max(0, int(np.floor(left - bw * margin)))
    right = min(w, int(np.ceil(right + bw * margin)))
    upper = max(0, int(np.floor(upper - bh * margin)))
    lower = min(h, int(np.ceil(lower + bh * margin)))
    if right - left < min_size:
        pad = int(np.ceil((min_size - (right - left)) / 2))
        left = max(0, left - pad)
        right = min(w, right + pad)
    if lower - upper < min_size:
        pad = int(np.ceil((min_size - (lower - upper)) / 2))
        upper = max(0, upper - pad)
        lower = min(h, lower + pad)
    return left, upper, right, lower


def adjust_camera(camera: Camera, new_id: int, crop: tuple[int, int, int, int]) -> Camera:
    left, upper, right, lower = crop
    params = camera.params.copy()
    _, _, cxcy = CAMERA_MODELS[camera.model_id]
    params[cxcy[0]] -= left
    params[cxcy[1]] -= upper
    return Camera(new_id, camera.model_id, right - left, lower - upper, params)


def make_preview(image_path: Path, crop: tuple[int, int, int, int], preview_path: Path) -> None:
    with Image.open(image_path) as im:
        im.thumbnail((512, 512))
        sx = im.width / Image.open(image_path).width
        sy = im.height / Image.open(image_path).height
        box = tuple(int(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(crop))
        draw = ImageDraw.Draw(im)
        draw.rectangle(box, outline=(255, 0, 0), width=3)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(preview_path)


def copy_sparse_extras(src_sparse: Path, dst_sparse0: Path) -> None:
    for item in src_sparse.iterdir():
        if item.name in {"cameras.bin", "images.bin"}:
            continue
        dst = dst_sparse0 / item.name
        if item.is_file():
            shutil.copy2(item, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--images", default="images")
    parser.add_argument("--percentile", default=1.0, type=float)
    parser.add_argument("--margin", default=0.15, type=float)
    parser.add_argument("--driver", choices=["colmap", "green", "hybrid"], default="hybrid")
    parser.add_argument("--min-mask-ratio", default=0.001, type=float)
    parser.add_argument("--max-mask-ratio", default=0.60, type=float)
    parser.add_argument("--min-points", default=30, type=int)
    parser.add_argument("--min-size", default=256, type=int)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    src_sparse = find_sparse_dir(source)
    src_images = source / args.images
    if not src_images.is_dir():
        raise FileNotFoundError(f"Images directory not found: {src_images}")

    cameras = read_cameras_binary(src_sparse / "cameras.bin")
    images = read_images_binary(src_sparse / "images.bin")

    dst_images = output / args.images
    dst_sparse0 = output / "sparse" / "0"
    dst_images.mkdir(parents=True)
    dst_sparse0.mkdir(parents=True)

    new_cameras: dict[int, Camera] = {}
    rows = []
    manifest = {
        "source": str(source),
        "output": str(output),
        "sparse_source": str(src_sparse),
        "images_dir": args.images,
        "crop_rule": {
            "driver": "registered_colmap_2d_points",
            "percentile": args.percentile,
            "margin": args.margin,
            "min_points": args.min_points,
            "min_size": args.min_size,
        },
        "images": {},
    }

    for image in images.values():
        src_image_path = src_images / image.name
        if not src_image_path.exists():
            raise FileNotFoundError(f"Image listed by COLMAP is missing: {src_image_path}")
        with Image.open(src_image_path) as im:
            width, height = im.size
            valid_xys = image.xys[image.point3d_ids != -1]
            crop_driver = args.driver
            crop_error = None
            crop = None
            if args.driver in {"green", "hybrid"}:
                try:
                    crop = crop_from_green_foreground(
                        im,
                        args.margin,
                        args.min_size,
                        args.min_mask_ratio,
                        args.max_mask_ratio,
                    )
                    crop_driver = "green"
                except Exception as exc:
                    crop_error = str(exc)
                    if args.driver == "green":
                        raise
            if crop is None:
                crop = crop_from_points(
                    valid_xys,
                    width,
                    height,
                    args.percentile,
                    args.margin,
                    args.min_points,
                    args.min_size,
                )
                crop_driver = "colmap"
            dst_image_path = dst_images / image.name
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            im.crop(crop).save(dst_image_path)

        if args.preview:
            make_preview(src_image_path, crop, output / "roi_preview" / f"{Path(image.name).stem}.jpg")

        new_camera_id = image.image_id
        old_camera = cameras[image.camera_id]
        new_cameras[new_camera_id] = adjust_camera(old_camera, new_camera_id, crop)
        left, upper, right, lower = crop
        image.camera_id = new_camera_id
        image.xys = image.xys.copy()
        image.xys[:, 0] -= left
        image.xys[:, 1] -= upper

        rows.append(
            {
                "image_id": image.image_id,
                "name": image.name,
                "old_camera_id": old_camera.camera_id,
                "new_camera_id": new_camera_id,
                "left": left,
                "top": upper,
                "right": right,
                "bottom": lower,
                "width": right - left,
                "height": lower - upper,
                "valid_colmap_points": int((image.point3d_ids != -1).sum()),
                "crop_driver": crop_driver,
                "fallback_reason": crop_error or "",
            }
        )
        manifest["images"][image.name] = rows[-1]

    write_cameras_binary(new_cameras, dst_sparse0 / "cameras.bin")
    write_images_binary(images, dst_sparse0 / "images.bin")
    copy_sparse_extras(src_sparse, dst_sparse0)

    with (output / "roi_crops.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "roi_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"ROI dataset written: {output}")
    print(f"Images: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
