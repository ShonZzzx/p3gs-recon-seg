from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from add_rgb_to_ply import read_header, make_dtype


def replacement_value(name: str, finite_values: np.ndarray) -> np.float32:
    if finite_values.size:
        return np.float32(np.median(finite_values))
    if name.startswith('rot_'):
        return np.float32(1.0 if name == 'rot_0' else 0.0)
    return np.float32(0.0)


def collect_replacements(src: Path, dtype: np.dtype, props: list[str], header_size: int, vertex_count: int, chunk_vertices: int) -> tuple[dict[str, np.float32], dict[str, int]]:
    float_fields = [name for name in props if dtype[name].kind == 'f']
    finite_samples: dict[str, list[np.ndarray]] = {name: [] for name in float_fields}
    bad_counts: dict[str, int] = {name: 0 for name in float_fields}
    max_samples_per_field = 1_000_000

    with src.open('rb') as fin:
        fin.seek(header_size)
        remaining = vertex_count
        while remaining:
            count = min(remaining, chunk_vertices)
            data = np.fromfile(fin, dtype=dtype, count=count)
            if data.size != count:
                raise ValueError(f'{src} ended early while scanning vertices')
            for name in float_fields:
                values = data[name]
                finite = np.isfinite(values)
                bad_counts[name] += int((~finite).sum())
                if bad_counts[name] and sum(x.size for x in finite_samples[name]) < max_samples_per_field:
                    need = max_samples_per_field - sum(x.size for x in finite_samples[name])
                    finite_samples[name].append(values[finite][:need].astype(np.float32, copy=True))
            remaining -= data.size

    replacements = {}
    for name in float_fields:
        if bad_counts[name]:
            samples = np.concatenate(finite_samples[name]) if finite_samples[name] else np.array([], dtype=np.float32)
            replacements[name] = replacement_value(name, samples)
    return replacements, {k: v for k, v in bad_counts.items() if v}


def convert_one(src: Path, dst: Path, overwrite: bool = False, chunk_vertices: int = 500_000) -> dict[str, int]:
    header, props, vertex_count, header_size = read_header(src)
    dtype = make_dtype(props, header)

    if dst.exists() and not overwrite:
        raise FileExistsError(f'{dst} already exists; pass --overwrite to replace it')

    replacements, bad_counts = collect_replacements(src, dtype, props, header_size, vertex_count, chunk_vertices)

    with src.open('rb') as fin, dst.open('wb') as fout:
        fin.seek(header_size)
        fout.write(header)
        remaining = vertex_count
        while remaining:
            count = min(remaining, chunk_vertices)
            data = np.fromfile(fin, dtype=dtype, count=count)
            if data.size != count:
                raise ValueError(f'{src} ended early while writing vertices')
            for name, value in replacements.items():
                bad = ~np.isfinite(data[name])
                if bad.any():
                    data[name][bad] = value
            data.tofile(fout)
            remaining -= data.size

    return bad_counts


def main() -> None:
    parser = argparse.ArgumentParser(description='Keep all 3DGS PLY properties and replace NaN/Inf float values.')
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('segmentData_rgb_fixed'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for src in args.inputs:
        dst = args.output_dir / src.name
        bad_counts = convert_one(src, dst, overwrite=args.overwrite)
        print(f'{src} -> {dst} fixed={bad_counts}', flush=True)


if __name__ == '__main__':
    main()