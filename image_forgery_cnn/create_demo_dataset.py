from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset for pipeline smoke tests.")
    parser.add_argument("--output", default="demo_data")
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_base_image(size: int, rng: random.Random) -> Image.Image:
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 0.35 + 0.35 * np.sin((grid_x * rng.uniform(3, 9) + rng.random()) * np.pi)
    base[..., 1] = 0.35 + 0.35 * np.cos((grid_y * rng.uniform(3, 9) + rng.random()) * np.pi)
    base[..., 2] = 0.45 + 0.25 * np.sin(((grid_x + grid_y) * rng.uniform(2, 7)) * np.pi)
    noise = np.random.default_rng(rng.randint(0, 10_000)).normal(0, 0.035, base.shape)
    image = np.clip(base + noise, 0, 1)
    return Image.fromarray((image * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=0.45))


def add_forgery_patch(image: Image.Image, rng: random.Random) -> Image.Image:
    forged = image.copy()
    draw = ImageDraw.Draw(forged)
    width, height = forged.size
    patch_w = rng.randint(width // 5, width // 3)
    patch_h = rng.randint(height // 5, height // 3)
    x0 = rng.randint(18, width - patch_w - 18)
    y0 = rng.randint(18, height - patch_h - 18)
    x1 = min(width, x0 + patch_w)
    y1 = min(height, y0 + patch_h)
    crop_x = rng.randint(0, width - patch_w)
    crop_y = rng.randint(0, height - patch_h)
    patch = image.crop((crop_x, crop_y, crop_x + patch_w, crop_y + patch_h))
    patch = patch.rotate(rng.choice([-3, -2, 2, 3]), resample=Image.Resampling.BICUBIC)
    forged.paste(patch, (x0, y0))
    outline = (rng.randint(185, 255), rng.randint(40, 100), rng.randint(40, 100))
    draw.rectangle((x0, y0, x1, y1), outline=outline, width=rng.randint(2, 4))
    return forged


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    root = Path(args.output).resolve()
    original_dir = root / "original"
    forged_dir = root / "forged"
    original_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.per_class):
        image = make_base_image(args.image_size, rng)
        image.save(original_dir / f"original_{idx:04d}.jpg", quality=92)
        forged = add_forgery_patch(image, rng)
        forged.save(forged_dir / f"forged_{idx:04d}.jpg", quality=92)

    print(f"Created demo dataset at: {root}")
    print(f"Original images: {args.per_class}")
    print(f"Forged images: {args.per_class}")


if __name__ == "__main__":
    main()
