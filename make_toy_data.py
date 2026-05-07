from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny synthetic images for a smoke test.")
    parser.add_argument("--out", default="data/toy_hr")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    yy, xx = np.mgrid[0 : args.size, 0 : args.size]
    for index in range(args.count):
        freq = rng.uniform(2.0, 10.0, size=3)
        phase = rng.uniform(0.0, np.pi * 2.0, size=3)
        image = np.zeros((args.size, args.size, 3), dtype=np.float32)
        for channel in range(3):
            waves = (
                np.sin((xx / args.size) * np.pi * freq[channel] + phase[channel])
                + np.cos((yy / args.size) * np.pi * (freq[channel] + 1.5) + phase[channel])
            )
            image[..., channel] = waves
        checker = ((xx // 16 + yy // 16 + index) % 2).astype(np.float32)
        image = 0.35 * image + 0.3 * checker[..., None] + rng.normal(0.0, 0.04, image.shape)
        image = (image - image.min()) / max(image.max() - image.min(), 1e-6)
        Image.fromarray((image * 255.0).astype(np.uint8)).save(out / f"toy_{index:03d}.png")

    print(f"wrote {args.count} images to {out}")


if __name__ == "__main__":
    main()

