from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from train_forgery_cnn import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    build_model,
    build_transforms,
    load_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run forgery detector on one image or a folder.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt or latest_model.pt.")
    parser.add_argument("--input", required=True, help="Image file or directory with images.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV with predictions.")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    images = [
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found in {path}")
    return sorted(images)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    image_size = int(config.get("image_size", 224))
    input_mode = str(config.get("input_mode", "rgb"))
    ela_quality = int(config.get("ela_quality", 90))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    transform = build_transforms(image_size, train=False)
    rows: list[dict[str, object]] = []

    for image_path in iter_images(input_path):
        image = load_image(image_path, input_mode=input_mode, ela_quality=ela_quality)
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor).squeeze(0).squeeze(0)
            probability = float(torch.sigmoid(logit).detach().cpu().item())
        label = 1 if probability >= args.threshold else 0
        row = {
            "path": str(image_path),
            "probability_forged": probability,
            "prediction": CLASS_NAMES[label],
        }
        rows.append(row)
        print(f"{image_path}: {row['prediction']} ({probability:.4f})")

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "probability_forged", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
