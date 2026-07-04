from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def read_header(path: Path) -> tuple[bytes, list[str], int, int]:
    with path.open("rb") as f:
        header = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            header.extend(line)
            if line.strip() == b"end_header":
                break

    text = header.decode("ascii")
    vertex_count = None
    properties: list[str] = []
    in_vertex = False

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ["format", "binary_little_endian"]:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and parts[0] == "property":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError(f"unsupported vertex property line: {line}")
            properties.append(parts[2])

    if vertex_count is None:
        raise ValueError(f"{path} has no vertex element")

    return bytes(header), properties, vertex_count, len(header)


def make_dtype(properties: list[str], header: bytes) -> np.dtype:
    prop_types: list[str] = []
    for line in header.decode("ascii").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "property":
            prop_types.append(parts[1])
            if len(prop_types) == len(properties):
                break

    if len(prop_types) != len(properties):
        raise ValueError("could not parse all vertex property types")

    fields = []
    for name, ply_type in zip(properties, prop_types):
        if ply_type not in PLY_TYPE_TO_DTYPE:
            raise ValueError(f"unsupported PLY type: {ply_type}")
        fields.append((name, PLY_TYPE_TO_DTYPE[ply_type]))
    return np.dtype(fields)


def add_rgb_properties(header: bytes) -> bytes:
    lines = header.decode("ascii").splitlines(keepends=True)
    out: list[str] = []
    inserted = False
    for line in lines:
        if line.strip() == "end_header" and not inserted:
            out.extend(
                [
                    "property uchar red\n",
                    "property uchar green\n",
                    "property uchar blue\n",
                ]
            )
            inserted = True
        out.append(line)
    return "".join(out).encode("ascii")


def convert_one(src: Path, dst: Path, overwrite: bool = False, chunk_vertices: int = 500_000) -> None:
    header, properties, vertex_count, header_size = read_header(src)
    if {"red", "green", "blue"}.issubset(properties):
        raise ValueError(f"{src} already has red/green/blue properties")
    for name in ("f_dc_0", "f_dc_1", "f_dc_2"):
        if name not in properties:
            raise ValueError(f"{src} is missing {name}")

    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")

    dtype = make_dtype(properties, header)
    out_dtype = np.dtype(dtype.descr + [("red", "u1"), ("green", "u1"), ("blue", "u1")])
    sh_c0 = 0.28209479177387814

    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(header_size)
        fout.write(add_rgb_properties(header))

        remaining = vertex_count
        while remaining:
            count = min(remaining, chunk_vertices)
            data = np.fromfile(fin, dtype=dtype, count=count)
            if data.size != count:
                raise ValueError(f"{src} ended early while reading vertices")

            out = np.empty(data.size, dtype=out_dtype)
            for name in properties:
                out[name] = data[name]

            rgb_float = np.stack(
                [data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]],
                axis=1,
            )
            rgb = np.clip((rgb_float * sh_c0 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
            out["red"] = rgb[:, 0]
            out["green"] = rgb[:, 1]
            out["blue"] = rgb[:, 2]
            out.tofile(fout)
            remaining -= data.size


def main() -> None:
    parser = argparse.ArgumentParser(description="Append standard red/green/blue properties to 3DGS PLY files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, help="directory for generated PLY files")
    parser.add_argument("--suffix", default="_rgb", help="suffix for generated PLY files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for src in args.inputs:
        if args.output_dir:
            dst = args.output_dir / f"{src.stem}{args.suffix}{src.suffix}"
        else:
            dst = src.with_name(f"{src.stem}{args.suffix}{src.suffix}")
        print(f"{src} -> {dst}", flush=True)
        convert_one(src, dst, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
