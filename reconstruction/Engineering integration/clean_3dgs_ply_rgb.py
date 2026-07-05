import argparse
import struct
from pathlib import Path


TYPE_FORMAT = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def read_header(handle):
    header_lines = []
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("PLY header is incomplete: missing end_header")
        text = line.decode("ascii", errors="replace").strip()
        header_lines.append(text)
        if text == "end_header":
            return header_lines


def parse_vertex_layout(header_lines):
    if "format binary_little_endian 1.0" in header_lines:
        ply_format = "binary_little_endian"
    elif "format ascii 1.0" in header_lines:
        ply_format = "ascii"
    else:
        raise ValueError("Only ascii and binary_little_endian PLY are supported")

    vertex_count = None
    properties = []
    in_vertex = False

    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue

        if in_vertex and len(parts) == 3 and parts[0] == "property":
            prop_type, prop_name = parts[1], parts[2]
            if prop_type not in TYPE_FORMAT:
                raise ValueError(f"Unsupported PLY property type: {prop_type}")
            properties.append((prop_name, prop_type))

    if vertex_count is None:
        raise ValueError("PLY has no vertex element")

    names = [name for name, _ in properties]
    for required in ("x", "y", "z", "red", "green", "blue"):
        if required not in names:
            raise ValueError(f"Missing required vertex property: {required}")

    offsets = {}
    offset = 0
    for name, prop_type in properties:
        fmt, size = TYPE_FORMAT[prop_type]
        offsets[name] = (offset, fmt, size)
        offset += size

    return ply_format, vertex_count, properties, offsets, offset


def clamp_color(value):
    if isinstance(value, float):
        value = round(value)
    return max(0, min(255, int(value)))


def write_output(output_path, records, keep_label=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as dst:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(records)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
        )
        if keep_label:
            header += "property float scalar_label\n"
        header += "end_header\n"
        dst.write(header.encode("ascii"))
        for record in records:
            dst.write(record)


def clean_ply(input_path, output_path, allow_truncated=False, keep_label=False):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("rb") as src:
        header = read_header(src)
        ply_format, vertex_count, properties, offsets, stride = parse_vertex_layout(header)
        property_names = [name for name, _ in properties]
        has_label = "scalar_label" in property_names
        if keep_label and not has_label:
            raise ValueError("Input PLY has no scalar_label property")

        clean_records = []

        if ply_format == "binary_little_endian":
            for index in range(vertex_count):
                record = src.read(stride)
                if len(record) != stride:
                    if allow_truncated:
                        break
                    raise ValueError(f"Unexpected EOF at vertex {index}")

                values = {}
                for name in ("x", "y", "z", "red", "green", "blue"):
                    offset, fmt, size = offsets[name]
                    values[name] = struct.unpack_from("<" + fmt, record, offset)[0]

                fmt = "<fffBBB"
                packed_values = [
                    float(values["x"]),
                    float(values["y"]),
                    float(values["z"]),
                    clamp_color(values["red"]),
                    clamp_color(values["green"]),
                    clamp_color(values["blue"]),
                ]
                if keep_label:
                    offset, label_fmt, size = offsets["scalar_label"]
                    packed_values.append(float(struct.unpack_from("<" + label_fmt, record, offset)[0]))
                    fmt += "f"
                clean_records.append(struct.pack(fmt, *packed_values))
        else:
            required_index = {name: property_names.index(name) for name in ("x", "y", "z", "red", "green", "blue")}

            for index in range(vertex_count):
                line = src.readline()
                if not line:
                    if allow_truncated:
                        break
                    raise ValueError(f"Unexpected EOF at vertex {index}")
                parts = line.decode("ascii", errors="replace").strip().split()
                if len(parts) < len(properties):
                    if allow_truncated:
                        break
                    raise ValueError(f"Vertex {index} has {len(parts)} values, expected {len(properties)}")

                fmt = "<fffBBB"
                packed_values = [
                    float(parts[required_index["x"]]),
                    float(parts[required_index["y"]]),
                    float(parts[required_index["z"]]),
                    clamp_color(float(parts[required_index["red"]])),
                    clamp_color(float(parts[required_index["green"]])),
                    clamp_color(float(parts[required_index["blue"]])),
                ]
                if keep_label:
                    packed_values.append(float(parts[property_names.index("scalar_label")]))
                    fmt += "f"
                clean_records.append(struct.pack(fmt, *packed_values))

    write_output(output_path, clean_records, keep_label=keep_label)

    return len(clean_records)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 3DGS PLY into a plain xyz + RGB PLY for CloudCompare."
    )
    parser.add_argument("input", help="Input 3DGS PLY")
    parser.add_argument("output", help="Output plain RGB PLY")
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="Recover valid prefix from a PLY that was truncated by a failed export",
    )
    parser.add_argument(
        "--keep-scalar-label",
        action="store_true",
        help="Keep the scalar_label property when it exists in the input PLY",
    )
    args = parser.parse_args()

    count = clean_ply(
        args.input,
        args.output,
        allow_truncated=args.allow_truncated,
        keep_label=args.keep_scalar_label,
    )
    print(f"Wrote {count} vertices to {args.output}")


if __name__ == "__main__":
    main()
