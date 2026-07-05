import argparse
from pathlib import Path

import cv2


def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser(description="Extract stable frames for GoF plant reconstruction.")
    parser.add_argument("--video", default=r"D:\Sophomore_AIA\ML_course\表型参数\大豆四输入视频.mp4")
    parser.add_argument("--out", default=r"D:\Sophomore_AIA\ML_course\soybean4_gof_frames_150\images")
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--end", type=int, default=980)
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument("--blur-threshold", type=float, default=80.0)
    args = parser.parse_args()

    video = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(args.end, total - 1)
    start = max(args.start, 0)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total}")

    count = args.target
    indices = [round(start + i * (end - start) / max(count - 1, 1)) for i in range(count)]

    saved = 0
    skipped_blur = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        score = blur_score(frame)
        if score < args.blur_threshold:
            skipped_blur += 1
            continue

        ok_write = cv2.imwrite(str(out_dir / f"frame_{idx:05d}.jpg"), frame)
        if not ok_write:
            raise RuntimeError(f"Failed to write frame {idx} to {out_dir}")
        saved += 1

    print(f"video={video}")
    print(f"total_frames={total}")
    print(f"range={start}-{end}")
    print(f"target={args.target}")
    print(f"saved={saved}")
    print(f"skipped_blur={skipped_blur}")
    print(f"out={out_dir}")


if __name__ == "__main__":
    main()
