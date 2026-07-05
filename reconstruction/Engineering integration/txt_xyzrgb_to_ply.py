import argparse
import struct
from pathlib import Path


def clamp_color(value):
    return max(0, min(255, int(round(float(value)))))


def parse_txt(input_path):
    records = []
    with Path(input_path).open("r", encoding="utf-8", errors="replace") as src:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.replace(",", " ").split()
            if len(parts) < 6:
                raise ValueError(f"Line {line_number} has {len(parts)} columns, expected at least 6")

            x, y, z = map(float, parts[:3])
            r, g, b = (clamp_color(parts[3]), clamp_color(parts[4]), clamp_color(parts[5]))
            records.append(struct.pack("<fffBBB", x, y, z, r, g, b))

    return records


def write_ply(output_path, records):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as dst:
        dst.write(
            (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(records)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            ).encode("ascii")
        )
        for record in records:
            dst.write(record)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a CloudCompare txt/asc export into a clean xyz + RGB PLY."
    )
    parser.add_argument("input", help="Input txt/asc file. Columns 1-3 must be xyz, columns 4-6 must be RGB.")
    parser.add_argument("output", help="Output clean PLY file")
    args = parser.parse_args()

    records = parse_txt(args.input)
    write_ply(args.output, records)
    print(f"Wrote {len(records)} vertices to {args.output}")


if __name__ == "__main__":
    main()
